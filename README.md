# MP-Mem-Instance-Sharing

* Author: CongVM

## Introduction

Process Module in Python has been forked when invoked by main process. It means they have theirs own memory, instances which are not shared between processes.

This module helps to overcome this limitation by using the `BaseManager` in `multiprocessing`.

In details, BaseManager creates a minimal TCP works as data transfer between processes.

