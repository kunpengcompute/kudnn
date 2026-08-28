import sys
import os

benchdnn_keyword = '/benchdnn/benchdnn'
result_keywords = ['passed:1', 'skipped:1', 'mistrusted:1', 'unimplemented:1', 'invalid_arguments:1', 'failed:1', 'listed:1']
avg_time_keyword = 'avg(ms):'
impl_keyword = 'implementation: '

def get_mode_p_data(mode_p_file_path):
    mode_p_file = open(mode_p_file_path, 'r')
    csv_p_data = []

    line = mode_p_file.readline()
    while line:
        if benchdnn_keyword in line:
            csv_p_data.append([])
            csv_p_data[-1].append(line[0:-1])
        if impl_keyword in line:
            pos = line.find(impl_keyword) + len(impl_keyword)
            csv_p_data[-1].append(line[pos:-1])
        if avg_time_keyword in line:
            pos = line.find(avg_time_keyword) + len(avg_time_keyword)
            # skipped
            if line[pos:-1] == '0':
                csv_p_data[-1].append('')
            csv_p_data[-1].append(line[pos:-1])
        line = mode_p_file.readline()
    mode_p_file.close()
    return csv_p_data

def get_mode_c_data(mode_c_file_path):
    mode_c_file = open(mode_c_file_path, 'r')
    csv_c_data = []

    line = mode_c_file.readline()
    while line:
        if benchdnn_keyword in line:
            csv_c_data.append([])
            csv_c_data[-1].append(line[0:-1])
        if impl_keyword in line:
            pos = line.find(impl_keyword) + len(impl_keyword)
            csv_c_data[-1].append(line[pos:-1])
        for result_keyword in result_keywords:
            if result_keyword in line:
                # skipped
                if result_keyword == 'skipped:1':
                    csv_c_data[-1].append('')
                csv_c_data[-1].append(result_keyword[0:-2])
                break
        line = mode_c_file.readline()
    mode_c_file.close()
    return csv_c_data

def write_csv_data(csv_p_data, csv_c_data, csv_dir_path):
    p_title='mode=P command,mode=P impl,mode=P avg(ms)'
    c_title='mode=C command,mode=C impl,mode=C result'

    # only output p data
    if len(csv_p_data) != 0:
        csv_file_path = csv_dir_path + '/output_p.csv'
        csv_file = open(csv_file_path, 'a+')
        csv_file.write(p_title + '\n')
        for data in csv_p_data:
            for i in range(len(data)):
                csv_file.write('"' + data[i] + '"' + ',')
            csv_file.write("\n")
        csv_file.close()

    # only output c data
    if len(csv_c_data) != 0:
        csv_file_path = csv_dir_path + '/output_c.csv'
        csv_file = open(csv_file_path, 'a+')
        csv_file.write(c_title + '\n')
        for data in csv_c_data:
            for i in range(len(data)):
                csv_file.write('"' + data[i] + '"' + ',')
            csv_file.write("\n")
        csv_file.close()

    # output p and c data together
    if len(csv_p_data) != 0 and len(csv_c_data) != 0:
        if len(csv_p_data) != len(csv_c_data):
            print('Number of test case p and c is not equal.')
            return
        csv_file_path = csv_dir_path + '/output_total.csv'
        csv_file = open(csv_file_path, 'a+')
        csv_file.write(p_title + ',' + c_title + '\n')
        for id in range(len(csv_p_data)):
            data_p = csv_p_data[id]
            data_c = csv_c_data[id]
            for i in range(len(data_p)):
                csv_file.write('"' + data_p[i] + '"' + ',')
            for i in range(len(data_c)):
                csv_file.write('"' + data_c[i] + '"' + ',')
            csv_file.write('\n')
        csv_file.close()

def main(argv):
    dir_path = argv[1]

    mode_p_file_path = dir_path + '/mode_p.log'
    csv_p_data = get_mode_p_data(mode_p_file_path)

    mode_c_file_path = dir_path + '/mode_c.log'
    csv_c_data = get_mode_c_data(mode_c_file_path)

    csv_dir_path = dir_path
    write_csv_data(csv_p_data, csv_c_data, csv_dir_path)

if __name__ == '__main__':
    main(sys.argv)
