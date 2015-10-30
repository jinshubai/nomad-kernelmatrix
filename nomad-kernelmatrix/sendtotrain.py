#!/usr/bin/env python
import sys, subprocess, os, math, shutil
if len(sys.argv) != 4 and len(sys.argv) !=5:
    print('usage: {0} machinefile input_file stored_dir [dst_dir]'.format(sys.argv[0]))
    sys.exit(1)
machinefile_path, input_file_path, temp_dir = sys.argv[1:4]

machines = set()
for line in open(machinefile_path):
    machine = line.strip()
    if machine in machines:
        print('Error: duplicated machine {0}'.format(machine))
        sys.exit(1)
    machines.add(machine)
nr_machines = len(machines)

src_basename = os.path.basename(input_file_path)
if len(sys.argv) == 5:
    dst_path = sys.argv[4]
else:
    dst_path = '{0}.sub'.format(src_basename)

print('Sending subdata to each computing node......')    
for i, machine in enumerate(machines):
    temp_path = os.path.join(temp_dir, src_basename + '.' + 
                             str(i))
    if machine == '127.0.0.1' or machine == 'master':
        cmd = 'mv {0} {1}'.format(temp_path,
                                  os.path.join('/home/jing/dis_data', dst_path))
    else:
        cmd = 'scp {0} {1}:{2}'.format(temp_path, machine,
                                       os.path.join('/home/jing/dis_data', dst_path))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    print('The subset of data has been copied to {0}'.format(machine))
shutil.rmtree(temp_dir)
