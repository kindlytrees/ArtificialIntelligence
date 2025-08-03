# Scheduling

- 进程调度发生在两个地方：1是进程挂起，2是定时器中断
- 进程调度的机制实现了量的swtch的上下文切换，一次是让出cpu运行的进程和内核调度器的切换，而是内核调度器和调度到的进程的切换


### 进程(线程)切换时的上下文有哪些

```c
// Saved registers for kernel context switches.
struct context {
  uint64 ra;
  uint64 sp;

  // callee-saved
  uint64 s0;
  uint64 s1;
  uint64 s2;
  uint64 s3;
  uint64 s4;
  uint64 s5;
  uint64 s6;
  uint64 s7;
  uint64 s8;
  uint64 s9;
  uint64 s10;
  uint64 s11;
};
```
### 上下文切换的汇编代码解析


```
a0 寄存器里存放的是一个指向旧上下文（old）结构体的指针。这是要被保存的上下文
a1 寄存器里存放的是一个指向新上下文（new）结构体的指针。这是要被加载的上下文
s0 - s11 (Callee-Saved Registers)：根据RISC-V调用约定，这些寄存器被称为“被调用者保存”（callee-saved）寄存器
.globl swtch
swtch:
        sd ra, 0(a0)  //保存返回地址 (Return Address) 寄存器 ra 到 old->ra,保存它至关重要，因为当这个旧上下文未来被恢复时，程序需要知道从哪里继续执行。
        sd sp, 8(a0)  // 保存栈指针 (Stack Pointer) 寄存器 sp 到 old->sp
        sd s0, 16(a0)
        sd s1, 24(a0)
        sd s2, 32(a0)
        sd s3, 40(a0)
        sd s4, 48(a0)
        sd s5, 56(a0)
        sd s6, 64(a0)
        sd s7, 72(a0)
        sd s8, 80(a0)
        sd s9, 88(a0)
        sd s10, 96(a0)
        sd s11, 104(a0)

        ld ra, 0(a1)
        ld sp, 8(a1)
        ld s0, 16(a1)
        ld s1, 24(a1)
        ld s2, 32(a1)
        ld s3, 40(a1)
        ld s4, 48(a1)
        ld s5, 56(a1)
        ld s6, 64(a1)
        ld s7, 72(a1)
        ld s8, 80(a1)
        ld s9, 88(a1)
        ld s10, 96(a1)
        ld s11, 104(a1)
        
        ret

```

```c
void
sched(void)
{
  int intena;
  struct proc *p = myproc();

  if(!holding(&p->lock))
    panic("sched p->lock");
  if(mycpu()->noff != 1)
    panic("sched locks");
  if(p->state == RUNNING)
    panic("sched running");
  if(intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}
```