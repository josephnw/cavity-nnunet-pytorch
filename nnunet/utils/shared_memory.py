from threading import Timer

import time
import mmap
import os
if os.name == 'nt':
    from win32event import CreateMutex
    from win32api import CloseHandle, GetLastError
    from winerror import ERROR_ALREADY_EXISTS
    import win32event
    import pywintypes
from struct import pack, unpack

LOCK_NAME = "AIState"#"Medip"


def WaitUntilGetMutex():
    STANDARD_RIGHTS_REQUIRED = 0xF0000
    SYNCHRONIZE = 0x100000
    MUTANT_QUERY_STATE = 0x1

    MUTEX_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE |MUTANT_QUERY_STATE
    try:
        hMutex = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, LOCK_NAME)
    except pywintypes.error:
        return None
    
    wait_result = win32event.WaitForSingleObject(hMutex, win32event.INFINITE)
    if (wait_result == win32event.WAIT_OBJECT_0):
        return hMutex
    return None
    '''if rc == win32event.WAIT_FAILED:	
        return -1
	if rc == win32event.WAIT_TIMEOUT:
        try:
            win32process.TerminateProcess (subprocess, 0)					
        except pywintypes.error:
            return -3
        return -2'''
def GetMemory(shm):
    mutex = WaitUntilGetMutex()
    if mutex is None:
        return -2
        #data = -1
    else:
        shm.seek(0)
        data = unpack("b", shm[:])[0]
    if data == -1:
        data = -2
        shm.seek(0)
        shm.write(pack('b', data))
    if mutex:
        win32event.ReleaseMutex(mutex)
    return data
def GetAndUpdateMemory( shm, status ):
    "Update status in shared memory"
    "1-99: percentage"
    "-1: failed"
    "1024: finished"
    mutex = WaitUntilGetMutex()
    if mutex is None:
        return -2
        #data = -1
    else:
        shm.seek(0)
        data = unpack("b", shm[:])[0]
    if data == -1:
        status = -2
    shm.seek(0)
    shm.write(pack('b', status))
    if mutex:
        win32event.ReleaseMutex(mutex)
    return status

def UpdateMemory( shm, status ):
    "Update status in shared memory"
    "1-99: percentage"
    "-1: failed"
    "1024: finished"
    mutex = WaitUntilGetMutex()
    if mutex is None:
        return 0
    shm.seek(0)
    shm.write(pack('b', status))
    if mutex:
        win32event.ReleaseMutex(mutex)
    return 1
'''
def WaitUntilGetMutex():
    STANDARD_RIGHTS_REQUIRED = 0xF0000
    SYNCHRONIZE = 0x100000
    MUTANT_QUERY_STATE = 0x1

    MUTEX_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE |MUTANT_QUERY_STATE
    hMutex = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, LOCK_NAME)
    wait_result = win32event.WaitForSingleObject(hMutex, 20000)
    if (wait_result == win32event.WAIT_OBJECT_0):
        return hMutex
def GetMemory(shm):
    #mutex = WaitUntilGetMutex()
    STANDARD_RIGHTS_REQUIRED = 0xF0000
    SYNCHRONIZE = 0x100000
    MUTANT_QUERY_STATE = 0x1

    MUTEX_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE |MUTANT_QUERY_STATE
    mutex = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, LOCK_NAME)
    wait_result = win32event.WaitForSingleObject(mutex, 2000)
    if (wait_result == win32event.WAIT_OBJECT_0):
        shm.seek(0)
        data = unpack("b", shm[:])[0]
        win32event.ReleaseMutex(mutex)
        return data
    else:
        win32event.ReleaseMutex(mutex)
        return 0
def UpdateMemory( shm, status ):
    "Update status in shared memory"
    "1-99: percentage"
    "-1: failed"
    "1024: finished"
    #mutex = WaitUntilGetMutex()
    STANDARD_RIGHTS_REQUIRED = 0xF0000
    SYNCHRONIZE = 0x100000
    MUTANT_QUERY_STATE = 0x1

    MUTEX_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE |MUTANT_QUERY_STATE
    mutex = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, LOCK_NAME)
    wait_result = win32event.WaitForSingleObject(mutex, 2000)
    if (wait_result == win32event.WAIT_OBJECT_0):
        shm.seek(0)
        shm.write(pack('b', status))
    win32event.ReleaseMutex(mutex)
    '''
'''
def WaitUntilGetMutex():
    mutex = CreateMutex(None, False, LOCK_NAME)
    lasterror = GetLastError()
    while(lasterror == ERROR_ALREADY_EXISTS):
        if mutex:
            CloseHandle(mutex)
        print("Other process is accessing memory. Wait 0.5 seconds.")
        time.sleep(0.5)
        mutex = CreateMutex(None, False, LOCK_NAME)
        lasterror = GetLastError()
    return mutex
def GetMemory(shm):
    mutex = WaitUntilGetMutex()
    shm.seek(0)
    data = shm[:].decode().replace('\x00','')
    if data == '':
        data = 0
    else:
        data = int(data)
    if mutex:
        CloseHandle(mutex)
    return data
def UpdateMemory( shm, status ):
    "Update status in shared memory"
    "1-99: percentage"
    "-1: failed"
    "100: finished"
    mutex = WaitUntilGetMutex()
    shm.seek(0)
    shm.write(str(status).encode()+b'\x00')
    if mutex:
        CloseHandle(mutex)
'''
