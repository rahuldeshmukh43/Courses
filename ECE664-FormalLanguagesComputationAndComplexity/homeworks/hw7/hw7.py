#! /bin/python
import re
import sys

textfilename = sys.argv[1]

with open(textfilename,'r') as f: lines = f.readlines()

for i,line in enumerate(lines):
    # check if line matches regular expression
    # mix of dots and dashes
    #regex = re.compile(r'^[ \t]*d[ \t]*[.-]{,4}[ \t]*i[ \t]*[.-]{,4}[ \t]*a[ \t]*[.-]{,4}[ \t]*g[ \t]*[.-]{,4}[ \t]*r[ \t]*[.-]{,4}[ \t]*a[ \t]*$')
    # either dots or dashes
    regex = re.compile(r'^[ \t]*d[ \t]*([.]{,4}|[-]{,4})[ \t]*i[ \t]*([.]{,4}|[-]{,4})[ \t]*a[ \t]*([.]{,4}|[-]{,4})[ \t]*g[ \t]*([.]{,4}|[-]{,4})[ \t]*r[ \t]*([.]{,4}|[-]{,4})[ \t]*a[ \t]*$')

    match = regex.match(line[:-1])
    if match is not None:
        print('line number '+str(i+1)+' in file contains match: '+ match.group() +'\n')
        flag=True
        #break

if not flag: print('SPAM free mail!! ')
