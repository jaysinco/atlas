.section .data
output:
    .asciz "The value is %d\n"
values:
    .int 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
.section .text
.global asmPrintf
asmPrintf:
    subq $24, %rsp
    mov %r12, (%rsp)
    mov %rbx, 8(%rsp)
    mov $0, %r12
    lea values(%rip), %rbx
loop:
    mov (%rbx, %r12, 4), %eax
    lea output(%rip), %rdi
    mov %eax, %esi
    xor %eax, %eax
    call printf
    inc %r12
    cmp $11, %r12
    jl loop
    mov (%rsp), %r12
    mov 8(%rsp), %rbx
    addq $24, %rsp
    ret
