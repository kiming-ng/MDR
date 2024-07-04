import json
import argparse

parser = argparse.ArgumentParser(description='GetMixPrompt')
parser.add_argument('--src_prompt_file', required=True, type=str, default=None)
parser.add_argument('--tgt_prompt_file', required=True, type=str, default=None)
parser.add_argument('--lamda', type=float, required=True, default=0.9)
# config dict
args = vars(parser.parse_args())

with open(args['src_prompt_file']) as f:
    data = json.load(f)
    for i in range(len(data)):
        for ctx in data[i]['ctxs']:
            ctx['meta_data']["mix"] = args['lamda'] * ctx['meta_data']["eigen"] + (1-args['lamda']) * ctx['meta_data']["loss"]
        data[i]['ctxs'] = sorted(data[i]['ctxs'], key = lambda x: x['meta_data']['mix'])

with open(args['tgt_prompt_file'], "w") as writer:
    writer.write(json.dumps(data, indent=4) + "\n")