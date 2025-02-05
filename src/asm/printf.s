.intel_syntax noprefix
.section .data
output:
    .asciz "The value is %d\n"
values:
    .int 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
.section .text
.global asmPrintf
asmPrintf:
    sub rsp, 24
    mov [rsp], r12
    mov [rsp + 8], rbx
    xor r12, r12
    lea rbx, [values + rip]
loop:
    mov eax, [rbx + r12 * 4]
    lea rdi, [output + rip]
    mov esi, eax
    xor eax, eax
    call printf
    inc r12
    cmp r12, 11
    jl loop
    mov r12, [rsp]
    mov rbx, [rsp + 8]
    add rsp, 24
    ret
