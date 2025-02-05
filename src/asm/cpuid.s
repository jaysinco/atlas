.intel_syntax noprefix
.section .data
output:
    .ascii "The processor Vendor ID is 'xxxxxxxxxxxx'\n"
.section .text
.global asmCpuid
asmCpuid:
    sub rsp, 8
    mov [rsp], rbx
    mov eax, 0
    cpuid
    lea rdi, [output + rip]
    mov [rdi + 28], ebx
    mov [rdi + 32], edx
    mov [rdi + 36], ecx
    call printf
    mov rbx, [rsp]
    add rsp, 8
    ret
