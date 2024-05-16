#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import shutil
from pathlib import Path


class PPTXDownloader:

    def __init__(self):
        self.suffix = ".pptx"
        self.fromDir = None
        self.toDir = None

    def copy_files(self, src_dir: str, dst_dir: str, 
                   extensions: list[str]) -> None:
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)

        for src_path in src_dir.rglob('*'):
            if src_path.is_file() and src_path.suffix in extensions:
                dst_path = dst_dir / src_path.relative_to(src_dir)
                print(dst_path)
                os.makedirs(dst_path.parent, exist_ok=True)
                shutil.copy2(src_path, dst_path)


    def run(self, src_dir: str, dst_dir: str,
            extensions: list[str] = ['.pptx', 'ppt']) -> None:
        pptx = PPTXDownloader()
        pptx.copy_files(src_dir, dst_dir, extensions)

