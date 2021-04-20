import argparse
import os
import pandas


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_clients", type=int, help="num of clients", default=4) 
    parser.add_argument("--in_filename", type=str) 
    # parser.add_argument("--out_path", type=str)  
    parser.add_argument("--glue_task_name", type=str)  
    args = parser.parse_args()

    return args

# filename = 'my.tsv'

args = parameters()

if __name__ == "__main__":
    in_filename = args.in_filename
    # output_dir = args.output_dir
    glue_task_name = args.glue_task_name
    num_clients = args.number_clients    
# load
    # data = pandas.read_csv(in_filename, sep = '\t', index_col = 'index')
    data = pandas.read_csv(in_filename, sep = '\t', index_col = 'index', error_bad_lines=False)
    data = data.sample(frac = 1.0, random_state= 1) # random seed can be change
    # print(data)
    
    total_lines = data.shape[0]
    client_lines = round(total_lines / num_clients)
    # print(client_lines)
    for i in range(num_clients):
        if i < num_clients - 1:
            data_cur = data[i * client_lines : (i + 1) * client_lines]
        else:
            data_cur = data[i * client_lines : total_lines]
        # print('clint ' + str(i) + ' :')
        data_cur.reset_index(drop = True, inplace = True)
        # print(data_cur)
# save
        out_dir = os.path.join('client' + str(i), glue_task_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, in_filename)
        data_cur.to_csv(out_path, sep='\t', index_label = 'index')