.section .data
output:
    .asciz "The value is %d\n"
values:
    .int 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
.section .text
.global main
main:
    mov $0, %r12d
loop:
    mov values(, %r12d, 4), %eax

    # printf
    mov $output, %rdi
    mov %eax, %esi
    xor %eax, %eax
    call printf

    inc %r12d
    cmp $11, %r12d
    jne loop

    # exit
    mov $0, %rdi
    call exit
