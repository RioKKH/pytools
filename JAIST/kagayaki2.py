#!/usr/bin/env python
# -*- coding: utf-8 -*-

import toml
import traceback
from typing import Dict

import pexpect

PYTOOLDIR = "/home/rio/work/git/pytools"


def load(fin: str) -> Dict:
    with open(fin) as fr:
        dobj = toml.load(fr)
    return dobj


def main():
    try:
        dobj = load("/".join([PYTOOLDIR, "config.toml"]))

        child = pexpect.spawn(
            "ssh %s@%s -X"
            % (dobj["kagayaki_configure"]["uname"], dobj["kagayaki_configure"]["addrs"])
        )
        # child = pexpect.spawn('ssh %s@kagayaki -Y' % dobj['kagayaki_configure']['user'])
        child.expect("%s" % dobj["kagayaki_configure"]["uname"])
        child.sendline("%s" % dobj["kagayaki_configure"]["pword"])
        child.expect("%s" % dobj["kagayaki_configure"]["uname"])

        print(child.before.decode(encoding="utf-8"))
        print(child.after.decode(encoding="utf-8"))

        child.interact()

    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
