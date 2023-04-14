# Homework 5: Processing Data Streams

## Tasks

### Task 1: 

### Task 2: 

### Task 2: 


## Stream Generation

The file for generating random data streams can be found [here](https://drive.google.com/drive/folders/1o7yFtJtPYtFUOnlaqNT_hk1thsoNo9B4?usp=sharing).

There's a single file called ```blackbox.py```.
It exposes the `Blackbox` that can bes used to query for batches of streams of given size.

The implementation is quite straightforward:
```
import random
import sys

class BlackBox:

    def ask(self, file, num):
        lines = open(file,'r').readlines()
        users = [0 for i in range(num)]
        for i in range(num):
            users[i] = lines[random.randint(0, len(lines) - 1)].rstrip("\n")
        return users

if __name__ == '__main__':
    bx = BlackBox()
    # users = bx.ask()
    # print(users)
```


## Dataset

The dataset for this task can be found [here](https://drive.google.com/drive/folders/1o7yFtJtPYtFUOnlaqNT_hk1thsoNo9B4?usp=sharing).

This dataset has a single seed file for generating the data stream called ```users.txt```.
It contains a list of users to be used as the seed for the stream generation.

The first few lines of the files are given below:
```
lzlZwIpuSWXEnNS91wxjHw
XvLBr-9smbI0m_a7dXtB7w
QPT4Ud4H5sJVr68yXhoWFw
i5YitlHZpf0B3R0s_8NVuw
s4FoIXE_LSGviTHBe8dmcg
ZcsZdHLiJGVvDHVjeTYYnQ
h3p6aeVL7vrafSOM50SsCg
EbJMotYYkq-iq-v1u8wCYA
nnB0AE1Cxp_0154xkhXelw
XoEnrhtJc2pcdlQ09d8Oug
QDQTMYp2NocktWN5fHwfIg
Nq9WrJcCjQjHSoaUPZcPWg
SgYDjNCecPidsRB_su5-tw
AUWHIxgZuL2h4svVLdUZaA
jlCxOfVf_Ff4YgGov8Tm1g
9u9a9JakFNHZksptLKPUrw
NfE1uHFWzzMyXkgBeEuR1A
RiBVI6UgLjfpA4EQ1SWDzA
iK_Mfq3BLLxprw4Trgc4Bg
```