.section .data
output:
    .ascii "The processor Vendor ID is 'xxxxxxxxxxxx'\n"
.section .text
.global asmCpuid
asmCpuid:
    subq $8, %rsp
    mov %rbx, (%rsp)
    movl $0, %eax
    cpuid
    lea output(%rip), %rdi
    movl %ebx, 28(%rdi)
    movl %edx, 32(%rdi)
    movl %ecx, 36(%rdi)
    call printf
    mov (%rsp), %rbx
    addq $8, %rsp
    ret
