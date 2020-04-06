from pprint import pprint
import os
import re
import sys
import subprocess
import yaml

from albertrunner import CommandCombiner, ProcessHandler

BINS = {
    "dataset":  ["/usr/bin/python3", "-m",  "albert.create_pretraining_data"],
    "pretrain": ["/usr/bin/python3", "ALBERT/run_pretraining.py"],
    "eval":     ["/usr/bin/python3", "ALBERT/run_pretraining.py"]
}

if len(sys.argv) < 6:
    print("usage: autobert [config] [input] [test_input] [model_dir] [steps] [savesteps]")
    #print("need config file")
    sys.exit(1)



with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
    #pprint(cfg)

if "dataset" in sys.argv:
    print("\n-- Creating dataest --")
    args = BINS["dataset"][:]
    for key, val in cfg["dataset"].items():
        args.append(f"--{key}")
        if isinstance(val, str) and len(val) > 0 and val[0] == "#":
            val = cfg["common"][val[1:]]
        args.append(str(val))
    
    print("-- Running:\n   ", " ".join(args), "\n--\n")
    
    #proc = subprocess.Popen (args, shell=False)
    #proc.communicate()
    
    print("\n-- Dataest created--\n")



def options_from_dict(d):
    out = []
    for key, value in d.items():
        if isinstance(value, bool):
            out.append(f"--{key}={value}")
            continue
        if isinstance(value, str) and len(value) > 0 and value[0] == "#":
            value = cfg["common"][value[1:]]
        out.append(f"--{key}")
        out.append(str(value))
    return out


class Trainer(ProcessHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.global_stepPS = 0
        self.examplesPS = 0
        self.last_checkpoint = 0
        
    def handle_stderr(self, line):
        match = re.match(".* global_step/sec: ([0-9\\.]+)\\n", line)
        if match:
            self.global_stepPS = float(match.group(1))
            return ""
        match = re.match(".* examples/sec: ([0-9\\.]+)\\n", line)
        if match:
            self.examplesPS = float(match.group(1))
            return ""
        match = re.match(".* Saving checkpoints for ([0-9]+) into .*\\n", line)
        if match:
            self.last_checkpoint = int(match.group(1))
            return line
        
        return line
    
    #def handle_stdout(self, text):
        #return text
        
    def get_status(self, term):
        return term.green(f"{self.last_checkpoint}: "
                          f"global_step/sec: {self.global_stepPS: 6.2f} | "
                          f"examples/sec: {self.examplesPS: 6.2f}")


class Evaluator(ProcessHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.eval_step = 0
        self.eval_prog = "[0/0]"
        self.mlm_acc = 0
        self.order_acc = 0
        
    def handle_stderr(self, line):
        match = re.match(".*  masked_lm_accuracy = ([0-9\\.]+)\\n", line)
        if match:
            self.mlm_acc = float(match.group(1))
            return line
        match = re.match(".*  sentence_order_accuracy = ([0-9\\.]+)\\n", line)
        if match:
            self.order_acc = float(match.group(1))
            return line
        match = re.match(".*  global_step = ([0-9]+)\\n", line)
        if match:
            self.eval_step = int(match.group(1))
            return line
        match = re.match(".* Evaluation (\\[.*\\])\\n", line)
        if match:
            self.eval_prog = match.group(1)
            return ""
        return line
    
    #def handle_stdout(self, text):
        #return text
        
    def get_status(self, term):
        return term.yellow(f"{self.eval_step} {self.eval_prog}: "
                           f"mlm_acc: {self.mlm_acc:.4f} | "
                           f"order_acc: {self.order_acc:.4f}")


# usage: autobert [config] [input] [test_input] [model_dir] [steps] [savesteps]


basepath = "models_pre/tenten2"
cmds = []
precfg = cfg["pretrain"]
precfg["albert_config_file"] = f"configs/albert_config.json"
precfg["input_file"] = sys.argv[2]
precfg["output_dir"] = f"{sys.argv[4]}/output/"
precfg["export_dir"] = f"{sys.argv[4]}/export/"
precfg["num_train_steps"] = sys.argv[5]
precfg["save_checkpoints_steps"] = sys.argv[6]
precfg["num_warmup_steps"] = "500"

precfg["do_train"] = True
precfg["do_eval"] = False
cmds.append(Trainer(BINS["pretrain"] + options_from_dict(precfg),
                      {"CUDA_VISIBLE_DEVICES": "0"}))
 
pprint(precfg)

precfg["input_file"] = sys.argv[3]
precfg["do_train"] = False
precfg["do_eval"] = True
cmds.append(Evaluator(BINS["pretrain"] + options_from_dict(precfg),
                      {"CUDA_VISIBLE_DEVICES": "1", "TF_CPP_MIN_LOG_LEVEL": "3"}))

#cmds = [" ".join(cmd) for cmd in cmds]
#pprint(cmds)

with CommandCombiner(cmds) as comb:
    comb.join()
