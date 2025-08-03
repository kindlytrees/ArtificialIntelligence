# File System

```c

//#define BSIZE 1024  // block size
struct buf {
  int valid;   // has data been read from disk?
  int disk;    // does disk "own" buf?
  uint dev;
  uint blockno;
  struct sleeplock lock;
  uint refcnt;
  struct buf *prev; // LRU cache list
  struct buf *next;
  uchar data[BSIZE];
};
```

dirent中的inum和其对应的inode中的inum保持一致

fs.img存储了文件系统中的布局信息

