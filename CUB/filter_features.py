import pandas as pd

input_file = "/home/konghaoz/cbm/CUB_200_2011/attributes/attributes.txt"
output_file = "/home/konghaoz/cbm/CUB_200_2011/attributes/attributes_filtered.txt"

indices_to_keep =  [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

# Read the input file, filter the lines, and write to the output file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Get the index from the line (assuming it's the first element before the space)
        index = int(line.split()[0])
        if index in indices_to_keep:
            outfile.write(line)
            
input_file = "/home/konghaoz/cbm/CUB_200_2011/attributes/image_attribute_labels_new.txt"
df = pd.read_csv(input_file, sep = "\s+", header=None)
filtered_df = df[df[1].isin(indices_to_keep)]
output_file = "/home/konghaoz/cbm/CUB_200_2011/attributes/image_attribute_labels_filtered.txt"
filtered_df.to_csv(output_file, sep=',', header=False, index=False)

