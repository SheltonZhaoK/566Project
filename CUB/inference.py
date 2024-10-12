"""
Evaluate trained models on the official CUB test set
"""
import os, copy, sys, torch, joblib, argparse, random
import numpy as np
from sklearn.metrics import f1_score
from ray import tune
from scipy.stats import pearsonr
from configs.mi_estimator_config import MI_ESTIMATOR
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from CUB.mi_estimator import mi_estimator
from CUB.dataset import load_data, find_partition_indices_by_IG, find_partition_indices_byRandom
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from analysis import AverageMeter, multiclass_metric, accuracy, binary_accuracy, compute_pearsoncr
from torchsummary import summary
from CUB.new_iterative_norm import IterNorm
from CUB.auxiliary_loss import calculate_ortho_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"

K = [1, 3, 5] #top k class accuracies to compute

def eval(args, seed, MI_estimator):
    """
    Run inference using model (and model2 if bottleneck)
    Returns: (for notebook analysis)
    all_class_labels: flattened list of class labels for each image
    topk_class_outputs: array of top k class ids predicted for each image. Shape = size of test set * max(K)
    all_class_outputs: array of all logit outputs for class prediction, shape = N_TEST * N_CLASS
    all_attr_labels: flattened list of labels for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs: flatted list of attribute logits (after ReLU/ Sigmoid respectively) predicted for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    all_attr_outputs_sigmoid: flatted list of attribute logits predicted (after Sigmoid) for each attribute for each image (length = N_ATTRIBUTES * N_TEST)
    wrong_idx: image ids where the model got the wrong class prediction (to compare with other models)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if args.model_dir:
        model = torch.load(args.model_dir)
    else:
        model = None
    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None
    model.eval()
    model.cuda()

    if args.model_dir2:
        if 'rf' in args.model_dir2:
            model2 = joblib.load(args.model_dir2)
        else:
            model2 = torch.load(args.model_dir2)
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()
    else:
        model2 = None

    if args.reduce == "random":
        indices = find_partition_indices_byRandom(args.n_attributes, args)
    elif args.reduce == "ig":
        indices = find_partition_indices_by_IG(args.n_attributes, args)

    if args.use_attr:
        attr_acc_meter = [AverageMeter()]
        if args.feature_group_results:  # compute acc for each feature individually in addition to the overall accuracy
            for _ in range(args.n_attributes):
                attr_acc_meter.append(AverageMeter())
    else:
        attr_acc_meter = None

    class_acc_meter = []
    cross_correlation_meter = AverageMeter()
    mi_meter = AverageMeter()
    for j in range(len(K)):
        class_acc_meter.append(AverageMeter())

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)
    all_outputs, all_targets = [], []
    all_attr_labels, all_attr_outputs, all_attr_outputs_sigmoid, all_attr_outputs2 = [], [], [], []
    all_attr_outputs_wr, all_attr_outputs_sigmoid_wr = [], []
    all_class_labels, all_class_outputs, all_class_logits = [], [], []
    topk_class_labels, topk_class_outputs = [], []

    for data_idx, data in enumerate(loader):
        if args.use_attr:
            if args.no_img:  # A -> Y
                inputs, labels = data
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs).t().float()
                inputs = inputs.float()
                # inputs = torch.flatten(inputs, start_dim=1).float()
            else:
                inputs, labels, attr_labels = data
                if args.n_attributes < 112:
                    attr_labels = [attr_labels[index] for index in indices]
                attr_labels = torch.stack(attr_labels).t()  # N x 312
        else:  # simple finetune
            inputs, labels = data

        inputs_var = torch.autograd.Variable(inputs).cuda()
        labels_var = torch.autograd.Variable(labels).cuda()

        if args.attribute_group:
            outputs = []
            f = open(args.attribute_group, 'r')
            for line in f:
                attr_model = torch.load(line.strip())
                outputs.extend(attr_model(inputs_var))
        else:
            outputs = model(inputs_var)
            concepts = torch.cat(outputs[1:args.n_attributes+1], dim=1)
            if args.residue > 0:
                residuals = torch.cat(outputs[args.n_attributes+1:], dim=1)    
            if args.crossC:
                crossCorrelation = calculate_ortho_loss(concepts, residuals)
                cross_correlation_meter.update(crossCorrelation.data.cpu().numpy(), inputs.size(0))
            if args.mi:
                mi = MI_estimator(concepts, residuals)
                mi_meter.update(mi.data.cpu().numpy(), inputs.size(0))
            if args.negTTI > 0:
                tti_concepts = torch.cat(outputs[1:args.n_attributes+1], dim=1)
                for i in range(len(tti_concepts)): # iterate through data points
                    subset = random.sample(list(range(args.n_attributes)), int(args.n_attributes * args.negTTI))
                    for index in subset:
                        tti_concepts[i][index] = np.random.randn()
                new_concepts = torch.split(tti_concepts, 1 ,dim=1)
                if args.residue > 0:
                    stage2_inputs = torch.cat([new_concepts, residuals], dim=1)
                else:
                    stage2_inputs = tti_concepts
                outputs = [model.sec_model(stage2_inputs)]
                outputs.extend(torch.split(stage2_inputs, 1, dim = 1))

            if args.no_img:  # A -> Y
                class_outputs = outputs
            else:
                if args.bottleneck:
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs = outputs
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
                    if model2:
                        stage2_inputs = torch.cat(attr_outputs, dim=1)
                        class_outputs = model2(stage2_inputs)
                    else:  # for debugging bottleneck performance without running stage 2
                        class_outputs = torch.zeros([inputs.size(0), N_CLASSES],
                                                    dtype=torch.float64).cuda()  # ignore this
                else:  # cotraining, end2end
                    if args.use_relu:
                        attr_outputs = [torch.nn.ReLU()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                    elif args.use_sigmoid:
                        attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs[1:]]
                        attr_outputs_sigmoid = attr_outputs
                    else:
                        attr_outputs_wr = outputs[1:]
                        attr_outputs_sigmoid_wr = [torch.nn.Sigmoid()(o) for o in outputs[1:]]

                        attr_outputs = outputs[1:args.n_attributes+1]
                        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs[1:args.n_attributes+1]]
                    
                    class_outputs = outputs[0]
                
                for i in range(args.n_attributes):
                    acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
                    acc = acc.data.cpu().numpy()
                    attr_acc_meter[0].update(acc, inputs.size(0))
                    if args.feature_group_results:  # keep track of accuracy of individual attributes
                        attr_acc_meter[i + 1].update(acc, inputs.size(0))

                unsqueezed_list = []
                for i in range (len(attr_outputs_wr)):
                    if i < args.n_attributes:
                        unsqueezed_list.append(attr_outputs_wr[i].unsqueeze(1))
                    else:
                        unsqueezed_list.append(attr_outputs_wr[i].unsqueeze(2))
                attr_outputs_wr = torch.cat(unsqueezed_list, dim=1)

                attr_outputs_sigmoid_wr = torch.cat([o for o in attr_outputs_sigmoid_wr], dim=1)

                attr_outputs = torch.cat([o.unsqueeze(1) for o in attr_outputs], dim=1)
                attr_outputs_sigmoid = torch.cat([o for o in attr_outputs_sigmoid], dim=1)
                all_attr_outputs.extend(list(attr_outputs.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid.extend(list(attr_outputs_sigmoid.flatten().data.cpu().numpy()))
                all_attr_labels.extend(list(attr_labels.flatten().data.cpu().numpy()))

                all_attr_outputs_wr.extend(list(attr_outputs_wr.flatten().data.cpu().numpy()))
                all_attr_outputs_sigmoid_wr.extend(list(attr_outputs_sigmoid_wr.flatten().data.cpu().numpy()))

        _, topk_preds = class_outputs.topk(max(K), 1, True, True)
        _, preds = class_outputs.topk(1, 1, True, True)
        all_class_outputs.extend(list(preds.detach().cpu().numpy().flatten()))
        all_class_labels.extend(list(labels.data.cpu().numpy()))
        all_class_logits.extend(class_outputs.detach().cpu().numpy())
        topk_class_outputs.extend(topk_preds.detach().cpu().numpy())
        topk_class_labels.extend(labels.view(-1, 1).expand_as(preds))

        np.set_printoptions(threshold=sys.maxsize)
        class_acc = accuracy(class_outputs, labels, topk=K)  # only class prediction accuracy
        for m in range(len(class_acc_meter)):
            class_acc_meter[m].update(class_acc[m], inputs.size(0))

    all_class_logits = np.vstack(all_class_logits)
    topk_class_outputs = np.vstack(topk_class_outputs)
    topk_class_labels = np.vstack(topk_class_labels)
    wrong_idx = np.where(np.sum(topk_class_outputs == topk_class_labels, axis=1) == 0)[0]

    for j in range(len(K)):
        print('Average top %d class accuracy: %.5f' % (K[j], class_acc_meter[j].avg))

    if args.use_attr and not args.no_img:  # print some metrics for attribute prediction performance
        print('Average attribute accuracy: %.5f' % attr_acc_meter[0].avg)
        all_attr_outputs_int = np.array(all_attr_outputs_sigmoid) >= 0.5
        if args.feature_group_results:
            n = len(all_attr_labels)
            all_attr_acc, all_attr_f1 = [], []
            for i in range(args.n_attributes):
                acc_meter = attr_acc_meter[1 + i]
                attr_acc = float(acc_meter.avg)
                attr_preds = [all_attr_outputs_int[j] for j in range(n) if j % args.n_attributes == i]
                attr_labels = [all_attr_labels[j] for j in range(n) if j % args.n_attributes == i]
                attr_f1 = f1_score(attr_labels, attr_preds)
                all_attr_acc.append(attr_acc)
                all_attr_f1.append(attr_f1)

            bins = np.arange(0, 1.01, 0.1)
            acc_bin_ids = np.digitize(np.array(all_attr_acc) / 100.0, bins)
            acc_counts_per_bin = [np.sum(acc_bin_ids == (i + 1)) for i in range(len(bins))]
            f1_bin_ids = np.digitize(np.array(all_attr_f1), bins)
            f1_counts_per_bin = [np.sum(f1_bin_ids == (i + 1)) for i in range(len(bins))]
            print("Accuracy bins:")
            print(acc_counts_per_bin)
            print("F1 bins:")
            print(f1_counts_per_bin)
            np.savetxt(os.path.join(args.log_dir, 'concepts.txt'), f1_counts_per_bin)

        balanced_acc, report = multiclass_metric(all_attr_outputs_int, all_attr_labels)
        f1 = f1_score(all_attr_labels, all_attr_outputs_int)
        print("Total 1's predicted:", sum(np.array(all_attr_outputs_sigmoid) >= 0.5) / len(all_attr_outputs_sigmoid))
        print('Avg attribute balanced acc: %.5f' % (balanced_acc))
        print("Avg attribute F1 score: %.5f" % f1)
        print(report + '\n')
    MI_estimator = None
    return class_acc_meter, attr_acc_meter, cross_correlation_meter, mi_meter, all_class_labels, topk_class_outputs, all_class_logits, all_attr_labels, all_attr_outputs_wr, all_attr_outputs_sigmoid_wr, wrong_idx, all_attr_outputs2

def train_mi_estimator(args):
    if args.model_dir:
        model = torch.load(args.model_dir)
    else:
        model = None
    if not hasattr(model, 'use_relu'):
        if args.use_relu:
            model.use_relu = True
        else:
            model.use_relu = False
    if not hasattr(model, 'use_sigmoid'):
        if args.use_sigmoid:
            model.use_sigmoid = True
        else:
            model.use_sigmoid = False
    if not hasattr(model, 'cy_fc'):
        model.cy_fc = None

    model.cuda()
    model.eval()

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)

    miEstimator, optimizer = None, None
    mi_args = argparse.Namespace(**MI_ESTIMATOR().config)
    miEstimator = mi_estimator(args = mi_args, mi_type="club", concept_dim=args.n_attributes, residual_dim=args.residue).cuda()
    optimizer = torch.optim.Adam(miEstimator.get_parameters(), lr=0.001, betas=(0.5, 0.999))
    for epoch in range(20):
        train_loss_meter = AverageMeter()
        for data_idx, data in enumerate(loader):
            if args.use_attr:
                if args.no_img:  # A -> Y
                    inputs, labels = data
                    if isinstance(inputs, list):
                        inputs = torch.stack(inputs).t().float()
                    inputs = inputs.float()
                    # inputs = torch.flatten(inputs, start_dim=1).float()
                else:
                    inputs, labels, attr_labels = data
            else:  # simple finetune
                inputs, labels = data

            inputs_var = torch.autograd.Variable(inputs).cuda()

            outputs = model(inputs_var)
            concepts = torch.cat(outputs[1:args.n_attributes+1], dim=1)
            residuals = torch.cat(outputs[args.n_attributes+1:], dim=1)

            mi_estimator_loss = miEstimator.estimator_loss(concepts, residuals)
            optimizer.zero_grad()
            mi_estimator_loss.backward()
            optimizer.step()
            train_loss_meter.update(mi_estimator_loss.data.cpu().numpy(), inputs.size(0))
        print(f"Epoch {epoch}: loss {train_loss_meter.avg}")
    return miEstimator

def run_inference(args):
    os.chdir("/home/shelton/supervised-concept")
    print(args)
    if type(args) is dict:
        args = args["args"]
    y_results, c_results, cross_results, mi_results = [], [], [], []

    for i, model_dir in enumerate(args.model_dirs):
        args.model_dir = model_dir
        args.model_dir2 = args.model_dirs2[i] if args.model_dirs2 else None

        if args.mi:
            MI_estimator = train_mi_estimator(args)
        else:
            MI_estimator = None
        result = eval(args, i+1, MI_estimator)
        class_acc_meter, attr_acc_meter, cross_cross_correlation_meter, mi_meter = result[0], result[1], result[2], result[3]
        y_results.append(1 - class_acc_meter[0].avg[0].item() / 100.)
        cross_results.append(cross_cross_correlation_meter.avg)
        mi_results.append(mi_meter.avg)
        if attr_acc_meter is not None:
            c_results.append(1 - attr_acc_meter[0].avg.item() / 100.)
        else:
            c_results.append(-1)
    values = (np.mean(y_results), np.std(y_results), np.mean(c_results), np.std(c_results), 
              np.mean(cross_results), np.std(cross_results), np.mean(mi_results), np.std(mi_results))
    output_string = '%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' % values
    print_string = 'Error of y: %.4f +- %.4f, Error of C: %.4f +- %.4f, Cross Correlation: %.4f +- %.4f, Mutual Information: %.4f +- %.4f' % values
    print(print_string)
    if args.negTTI > 0:
        output = open(os.path.join(args.log_dir, 'negTTI.txt'), 'w')
    else:
        output = open(os.path.join(args.log_dir, 'results.txt'), 'w')
    output.write(output_string)

def create_args_list(args, search_space):
    arg_list = [copy.deepcopy(args) for _ in range(len(search_space))]

    for index, element in enumerate(search_space):
        arg_list[index].n_attributes = element
        arg_list[index].subset = element
        arg_list[index].log_dir = os.path.join(arg_list[index].log_dir, str(element))
        for pos, i in enumerate(range(1,2,1)):
            arg_list[index].model_dirs[pos] = os.path.join(arg_list[index].model_dirs[pos], str(element), f"best_model_{i}.pt")
    return arg_list

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-log_dir', default='.', help='where results are stored')
    parser.add_argument('-model_dirs', default=None, nargs='+', help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, nargs='+', help='where another trained model are saved (for bottleneck only)')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-bottleneck', help='whether to predict attributes before class labels', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-attribute_group', default=None, help='file listing the (trained) model directory for each attribute group')
    parser.add_argument('-feature_group_results', help='whether to print out performance of individual atttributes', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-subset', type = int, default = 112, help = 'Percentage of subset of concepts used')
    parser.add_argument('-disentangle', default = None, help = 'Vector disentanglement method')
    parser.add_argument('-reduce', default = None, help = 'whether the concepts are reduced by random or importance')
    parser.add_argument('-mi', action='store_true', help='if included, estimate the mi between concept and residual')
    parser.add_argument('-crossC', action='store_true', help='if included, compute the cross correlation between concept and residual')
    parser.add_argument('-residue', type = int, default = 0, help = 'number of residual used')
    parser.add_argument('-ray', action = "store_true", help="whether to use ray to perform inference")
    parser.add_argument('-negTTI', default=0.0, type = float, help="percentage of data to perform negative intervention")
    parser.add_argument('-dataset', default='cub', help="dataset")

    args = parser.parse_args()
    args.n_attributes = args.subset
    args.batch_size = 32
    if args.ray:
        arg_list = create_args_list(args, [20,40,60,80,100,112])
        # arg_list = create_args_list(args, [20])
        config = {"args": tune.grid_search(arg_list)}
        
        tuner = tune.run(
            tune.with_resources(tune.with_parameters(run_inference), {"cpu": 1, "gpu": 1}),
            config = config
        )
    else:
        run_inference(args)