.section .data
output:
    .ascii "The processor Vendor ID is 'xxxxxxxxxxxx'\n"
.section .text
.global main
main:
    movl $0, %eax
    cpuid
    movq $output, %rdi
    movl %ebx, 28(%rdi)
    movl %edx, 32(%rdi)
    movl %ecx, 36(%rdi)
    movq $1, %rax
    movl $1, %edi
    movq $output, %rsi
    movq $42, %rdx
    syscall
    movq $60, %rax
    movl $0, %edi
    syscall
