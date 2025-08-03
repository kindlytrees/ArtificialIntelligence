# Page Tables Notes

在引导序列的早期，main 调用 kvminit（kernel/vm.c:54）

```c
// Initialize the one kernel_pagetable
void
kvminit(void)
{
  kernel_pagetable = kvmmake();
}

```
此时，分页机制尚未启用，因此地址直接引用物理内存。  
kvmmake 首先分配一页物理内存来保存根页表，然后调用 kvmmap 安装内核需要的映射，这些映射包括内核的指令和数据、物理内存（直到 PHYSTOP），以及实际为设备预留的内存范围。  
通过 kvmmake（kernel/vm.c:20）来创建内核的页表。

```c
// Make a direct-map page table for the kernel.
pagetable_t
kvmmake(void)
{
  pagetable_t kpgtbl;

  kpgtbl = (pagetable_t) kalloc();
  memset(kpgtbl, 0, PGSIZE);

  // uart registers
  kvmmap(kpgtbl, UART0, UART0, PGSIZE, PTE_R | PTE_W);

  // virtio mmio disk interface
  kvmmap(kpgtbl, VIRTIO0, VIRTIO0, PGSIZE, PTE_R | PTE_W);

  // PLIC
  kvmmap(kpgtbl, PLIC, PLIC, 0x4000000, PTE_R | PTE_W);

  //kvmmap(pagetable_t kpgtbl, uint64 va, uint64 pa, uint64 sz, int perm)
  //map kernel text executable and read-only.
  kvmmap(kpgtbl, KERNBASE, KERNBASE, (uint64)etext-KERNBASE, PTE_R | PTE_X);

  // map kernel data and the physical RAM we'll make use of.
  //分配地址直至PHYSTOP
  kvmmap(kpgtbl, (uint64)etext, (uint64)etext, PHYSTOP-(uint64)etext, PTE_R | PTE_W);

  // map the trampoline for trap entry/exit to
  // the highest virtual address in the kernel.
  kvmmap(kpgtbl, TRAMPOLINE, (uint64)trampoline, PGSIZE, PTE_R | PTE_X);

  // allocate and map a kernel stack for each process.
  proc_mapstacks(kpgtbl);
  
  return kpgtbl;
}

```


进程栈的映射： proc_mapstacks（kernel/proc.c:33）为每个进程分配内核栈。它通过调用 kvmmap 将栈映射到由 KSTACK 生成的虚拟地址，从而为无效的栈保护页（guard page）留出空间。

```c
// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void
proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;
  
  for(p = proc; p < &proc[NPROC]; p++) {
    char *pa = kalloc();
    if(pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int) (p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}
```

映射函数： kvmmap（kernel/vm.c:132）

```c
// add a mapping to the kernel page table.
// only used when booting.
// does not flush TLB or enable paging.
void
kvmmap(pagetable_t kpgtbl, uint64 va, uint64 pa, uint64 sz, int perm)
{
  if(mappages(kpgtbl, va, sz, pa, perm) != 0)
    panic("kvmmap");
}

```

调用 mappages（kernel/vm.c:144）  
mappages 会依次为每个虚拟地址安装映射，按照页大小间隔逐一进行。对于每个需要映射的虚拟地址，mappages 调用 walk 查找该地址的 PTE 地址，然后初始化 PTE，设置对应的物理页号、所需的权限（PTE_W、PTE_X 和/或 PTE_R），以及通过 PTE_V 标记 PTE 为有效。


```c

// Create PTEs for virtual addresses starting at va that refer to
// physical addresses starting at pa.
// va and size MUST be page-aligned.
// Returns 0 on success, -1 if walk() couldn't
// allocate a needed page-table page.
int
mappages(pagetable_t pagetable, uint64 va, uint64 size, uint64 pa, int perm)
{
  uint64 a, last;
  pte_t *pte;

  if((va % PGSIZE) != 0)
    panic("mappages: va not aligned");

  if((size % PGSIZE) != 0)
    panic("mappages: size not aligned");

  if(size == 0)
    panic("mappages: size");
  
  a = va;
  last = va + size - PGSIZE;
  for(;;){
    if((pte = walk(pagetable, a, 1)) == 0)
      return -1;
    if(*pte & PTE_V)
      panic("mappages: remap");
    *pte = PA2PTE(pa) | perm | PTE_V;
    if(a == last)
      break;
    a += PGSIZE;
    pa += PGSIZE;
  }
  return 0;
}

```

查找 PTE（walk）： walk（kernel/vm.c:86）

```c
// Return the address of the PTE in page table pagetable
// that corresponds to virtual address va.  If alloc!=0,
// create any required page-table pages.
//
// The risc-v Sv39 scheme has three levels of page-table
// pages. A page-table page contains 512 64-bit PTEs.
// A 64-bit virtual address is split into five fields:
//   39..63 -- must be zero.
//   30..38 -- 9 bits of level-2 index.
//   21..29 -- 9 bits of level-1 index.
//   12..20 -- 9 bits of level-0 index.
//    0..11 -- 12 bits of byte offset within the page.
pte_t *
walk(pagetable_t pagetable, uint64 va, int alloc)
{
  if(va >= MAXVA)
    panic("walk");

  for(int level = 2; level > 0; level--) {
    pte_t *pte = &pagetable[PX(level, va)];
    if(*pte & PTE_V) {          	//92
      pagetable = (pagetable_t)PTE2PA(*pte);
    } else {
      if(!alloc || (pagetable = (pde_t*)kalloc()) == 0)
        return 0;
      memset(pagetable, 0, PGSIZE);
      *pte = PA2PTE(pagetable) | PTE_V;
    }
  }
  return &pagetable[PX(0, va)];		//102
}
```

上面的walk函数模拟了 RISC-V 分页硬件的工作原理，逐级查找虚拟地址的 PTE（参见图 3.2）。walk 会依次沿着页表层级查找，使用每一层的 9 位虚拟地址索引到相应的页目录页面。  
在每一层，它要么找到下一层页目录页面的 PTE，要么找到最终的页面的 PTE（kernel/vm.c:92）。如果页目录中的 PTE 无效，则说明该目录页面尚未分配；  
如果 alloc 参数被设置，walk 会分配一个新的页表页面，并将其物理地址填入 PTE 中。最终，walk 返回最低层页表中的 PTE 地址（kernel/vm.c:102）。  

内存的直接映射： 上面的代码依赖于物理内存直接映射到内核虚拟地址空间。例如，在 walk 函数中，它通过从 PTE 中取出下一层页表的物理地址（kernel/vm.c:94），然后将该地址当作虚拟地址，继续向下查找 PTE（kernel/vm.c:92）。这依赖于内核和物理内存之间的直接映射关系。

安装内核页表： main 调用 kvminithart（kernel/vm.c:62）来安装内核页表。它将根页表页面的物理地址写入 satp 寄存器，之后 CPU 会使用内核页表来翻译地址。由于内核使用直接映射，下一条指令的虚拟地址将直接映射到正确的物理地址。

```c
// Switch h/w page table register to the kernel's page table,
// and enable paging.
void
kvminithart()
{
  // wait for any previous writes to the page table memory to finish.
  sfence_vma();

  w_satp(MAKE_SATP(kernel_pagetable));

  // flush stale entries from the TLB.
  sfence_vma();
}

```