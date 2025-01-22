.section .data
msg:
    .ascii "stack %d: %p\n"
.section .text
.global main
main:
    pushq %rbp
    movq %rsp, %rbp
    call fun1
    movq $60, %rax
    movl $0, %edi
    syscall
fun1:
    pushq %rbp
    movq %rsp, %rbp
    call fun2
    movq %rbp, %rsp
    popq %rbp
    ret
fun2:
    pushq %rbp
    movq %rsp, %rbp
    call fun3
    movq %rbp, %rsp
    popq %rbp
    ret
fun3:
    pushq %rbp
    movq %rsp, %rbp
    call print_stack
    movq %rbp, %rsp
    popq %rbp
    ret
print_stack:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    movl $0, %r12d
    movq %rbp, %rbx
print_stack_loop:
    test %rbx, %rbx
    jz print_stack_done
    mov $msg, %rdi
    mov %r12d, %esi
    mov 8(%rbx), %rdx
    xor %eax, %eax
    call printf
    mov (%rbx), %rbx
    inc %r12d
    jmp print_stack_loop
print_stack_done:
    popq %r12
    popq %rbx
    movq %rbp, %rsp
    popq %rbp
    ret
