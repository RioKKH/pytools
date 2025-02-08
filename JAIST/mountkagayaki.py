#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import toml
import traceback
from typing import Dict

import pexpect

PYTOOLDIR="/home/rio/work/git/pytools"

def load(fin:str) -> Dict:
    with open(fin) as fr:
        dobj = toml.load(fr)
    return dobj

def main():
    try:
        dobj = load('/'.join([PYTOOLDIR, 'config.toml']))

        addrs = dobj['kagayaki_configure']['addrs']
        uname = dobj['kagayaki_configure']['uname']
        pword = dobj['kagayaki_configure']['pword']
        luser = dobj['kagayaki_configure']['luser']
        mount = dobj['kagayaki_configure']['mount']

        pstr = f'sshfs {uname}@{addrs}:/home/{uname} /home/{luser}/{mount}'
        #pstr = f'sshfs {uname}@{cmptr}:/home/{uname} /home/{luser}/{cmptr}'
        p = pexpect.spawn(pstr, ignore_sighup=True)

        p.expect(  "%s" % dobj['kagayaki_configure']['uname'])
        p.sendline("%s" % dobj['kagayaki_configure']['pword'])
        #print(p.before.decode(encoding='utf-8'))
        #print(p.after.decode(encoding='utf-8'))
        p.expect(pexpect.EOF)

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
