# Copyright 2020 Petr Zelina.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import threading
import subprocess
import blessings
import re
from collections import defaultdict


class ThreadReader(threading.Thread):
    def __init__(self, stream, combiner, i=0):
        threading.Thread.__init__(self)
        self.stream = stream
        self.combiner = combiner
        self.i = f"p{i}: "

    def run(self):
        for line in self.stream:
            #print(line.decode())
            self.combiner(self.i + line.decode())



class ProcessHandler:
    def __init__(self, cmd, env=None):
        self.cmd = cmd
        if env is None:
            self.env = {}
        else:
            self.env = env
    
    def handle_stderr(self, text):
        return text
    
    def handle_stdout(self, text):
        return text
    
    def get_status(self, term):
        return ""
    

class CommandCombiner:
    def __init__(self, handlers):
        self.term = blessings.Terminal()
        self.running = []
        self.readers = []
        self.handlers = handlers
        self.cmdid = 0

    def __enter__(self):
        for handler in self.handlers:
            self._run_handler(handler)
            
        return self
    
    def __exit__(self, *args):
        print()
        self.join()
        print()
    
    def _run_handler(self, handler):
        env = os.environ.copy()
        for key, val in handler.env.items():
            env[key] = val
        p = subprocess.Popen(handler.cmd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.running.append(p)
        
        std_writer, err_writer = self._bind_handler(handler)
        r = ThreadReader(p.stdout, std_writer, self.cmdid)
        r.start()
        self.readers.append(r)
        self.cmdid += 1
        r = ThreadReader(p.stderr, err_writer, self.cmdid)
        r.start()
        self.readers.append(r)
        self.cmdid += 1
        
    def _bind_handler(self, handler):
        def _print_stdout(line):
            line = handler.handle_stdout(line)
            if line is not None:
                self._print_line(line)
            
        def _print_stderr(line):
            line = handler.handle_stderr(line)
            if line is not None:
                self._print_line(line)
        
        return _print_stdout, _print_stderr
    
    def _get_status(self):
        status = [handler.get_status(self.term) for handler in self.handlers]
        return " || ".join(status)
    
    def _print_line(self, line):
        print(self.term.move_x(0), self.term.clear_eol, end="", sep="")
        print(line, end="")
        status = self._get_status()
        print(status, end="", flush=True)
    
    def join(self):
        #for reader in self.readers:
            #reader.start()
            #proc.start()
    
        for proc in self.running:
            proc.wait()


if __name__ == "__main__":
    cmds = ["for f in `/bin/ls`; do echo $f; sleep 1; done",
            "for f in {0..12}; do echo prog: $f.1; sleep 0.8; done"]

    #cmds = ["primes 1",
            #"primes 1"]

    with CommandCombiner(cmds) as comb:
        comb.join()
