# Homework 5: Processing Data Streams

In this assignment, on-line algorithms relating to the processing of data streams are explored.
Take a look at the [homework specs](Homework%205%20Description.pdf) for exact details.

## Tasks

### Task 1: 
Here the Bloom Filtering algorithm is implemented to estimate whether the user_id in the data stream has
been seen before or not.

### Task 2: 
Here the Flajolet-Martin algorithm (including the step of combining estimations from groups of hash functions) to 
estimate the number of unique users within a window in the data stream is implemented.


### Task 3: 
The goal of this task is to implement the fixed size sampling method (Reservoir Sampling Algorithm).
It is assumed that the memory can only save `100` users, so we need to use the fixed size
sampling method to only keep part of the users as a sample in the streaming. When the streaming of the
users start the first 100 users are directly saved in to the reservoir (a list). After that, for the nth (n starts
from 1) user in the whole sequence of users, the nth user is saved with the probability of 100/n 
and randomly replaced with one of the users in the list.

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