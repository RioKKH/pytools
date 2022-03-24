#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import toml
import traceback
from typing import Dict

import pexpect

def load(fin:str) -> Dict:
    with open(fin) as fr:
        dobj = toml.load(fr)
    return dobj

def main():
    try:
        dobj = load('config.toml')

        uname = dobj['configure']['user']
        pword = dobj['configure']['pass']
        #pstr = f'sshfs {uname}@kagayaki:/home/{uname} /home/kakehi/kagayaki'
        #p = pexpect.spawn('sshfs %s@kagayaki' % dobj['configure']['user'])
        p = pexpect.spawn('sshfs s2030025@kagayaki:/home/s2030025 /home/kakehi/kagayaki')
        print('hi1')

        p.expect("s2030025")
        print('hi2')
        p.sendline('S@wakkoliekkyo0824')
        print('hi3')
        #p.expect("s2030025")
        #p.expect("%s" % dobj['configure']['user'])
        #p.sendline('%s' % dobj['configure']['pass'])
        #p.expect("kakehi" % dobj['configure']['user'])
        p.expect("kakehi")
        p.interact()
        print('hi4')

        #p = pexpect.spawn(pstr)
        #p.expect("%s" % str(uname))
        #p.sendline("%s" % str(pword))
        #p.expect("%s" % str(uname))

        #p.expect(f"{uname}")
        #print(p.before.decode(encoding='utf-8'))
        #print(p.after.decode(encoding='utf-8'))


    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
