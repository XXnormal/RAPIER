import os
import sys
import dpkt
import socket

from xgboost import train
workspace=sys.path[0]

class one_flow(object):
    def __init__(self,pkt_id,timestamp,direction,pkt_length):
        #Fixed
        self.pkt_id = pkt_id
        
        detailed_info = pkt_id.split("_")
        self.client_ip = detailed_info[0]
        self.client_port = int(detailed_info[1])
        self.outside_ip = detailed_info[2]
        self.outside_port = int(detailed_info[3])
        
        self.start_time = timestamp
        #Updatable
        self.last_time = timestamp
        self.pkt_count = 1
        
        self.burst_list = [one_burst(timestamp, direction, pkt_length)]
    
    def update(self,timestamp,direction,pkt_length):
        self.pkt_count += 1
        self.last_time = timestamp
        
        if self.burst_list[-1].direction != direction:
            self.burst_list.append(one_burst(timestamp,direction,pkt_length))
        else:
            self.burst_list[-1].update(timestamp,pkt_length)
            
class one_burst(object):
    def __init__(self,timestamp,direction,pkt_length):
        #Fixed
        self.direction = direction
        self.start_time = timestamp
        #Updatable
        self.last_time = timestamp
        self.pkt_count = 1
        self.pkt_length = pkt_length
        
    def update(self,timestamp,pkt_length):
        self.last_time = timestamp
        
        self.pkt_count += 1
        self.pkt_length += pkt_length
		
def inet_to_str(inet):
	return socket.inet_ntop(socket.AF_INET, inet)

def get_burst_based_flows(pcap):
    current_flows = dict()
    for i, (timestamp, buf) in enumerate(pcap):
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception as e:
            print(e)
            continue
        
        if not isinstance(eth.data, dpkt.ip.IP):
            eth = dpkt.sll.SLL(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
		
        ip = eth.data
        pkt_length = ip.len
		
        src_ip = inet_to_str(ip.src)
        dst_ip = inet_to_str(ip.dst)
    
        if not isinstance(ip.data, dpkt.tcp.TCP):
            continue

        tcp = ip.data
        srcport = tcp.sport
        dstport = tcp.dport
        direction = None
        
        if dstport == 443:
            direction = -1
            pkt_id = src_ip+"_"+str(srcport)+"_"+dst_ip+"_"+str(dstport)
        elif srcport == 443:
            direction = 1
            pkt_id = dst_ip+"_"+str(dstport)+"_"+src_ip+"_"+str(srcport)
        else:
            continue
        
        if pkt_id in current_flows:
            current_flows[pkt_id].update(timestamp,direction,pkt_length)
        else:
            current_flows[pkt_id] = one_flow(pkt_id,timestamp,direction,pkt_length)

    return list(current_flows.values())

def get_flows(file):
    with open(file,"rb") as input:
        pcap = dpkt.pcap.Reader(input)
        all_flows = get_burst_based_flows(pcap)
        return all_flows

def generate_sequence_data(all_files_flows, output_file, label_file):
    output_features = []
    output_labels = []
    for flow in all_files_flows:
        one_flow = []
        client_ip = flow.client_ip
        outside_ip = flow.outside_ip
        label = client_ip + '-' + outside_ip
        for index,burst in enumerate(flow.burst_list):
            if index != 0:
                current_cumulative = one_flow[-1] + (burst.pkt_length * burst.direction)
                one_flow.append(current_cumulative)
            else:
                one_flow.append(burst.pkt_length * burst.direction)
        
        one_flow = [str(value) for value in one_flow]
        one_line = ",".join(one_flow)
        output_features.append(one_line)
        output_labels.append(label)
    
    write_into_files(output_features, output_file)
    write_into_files(output_labels, label_file)

def write_into_files(output_features,output_file):
    with open(output_file,"w") as write_fp:
        output_features = [value+"\n" for value in output_features]
        write_fp.writelines(output_features)

def main(input_dir, output_path, suffix):
    #Output feature files
    pcap_filedir = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file[-len(suffix)-1:] == '.'+suffix:
                pcap_filedir.append(os.path.join(root, file))

    files = pcap_filedir
    all_files_flows = []
    for file in files:
        try:
            flows_of_file=get_flows(file)
        except Exception as e:
            print(e)
            pass
        if flows_of_file==False:#错误记录
            print(file, "Critical Error2")
            continue
        if len(flows_of_file) <= 0:
            continue
        all_files_flows += flows_of_file
 
    generate_sequence_data(all_files_flows, output_path, output_path + '_labels')

if __name__ == "__main__":
	
    _, input_dir, output_path, suffix = sys.argv
    main(input_dir, output_path, suffix)
    
