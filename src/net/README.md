### Get SSH Host
```
sudo netstat -tnpa | grep sshd
```

### Find Ip or Mac
```
arp-scan --localnet | grep <mac or ip>
nmap -n -sn --send-ip 192.168.3.0/24
ip neigh | grep <mac or ip>
```

### Sample Mac
* a4:17:91:0f:fe:04 router
* ec:d6:8a:d6:7d:82 sinco
* ec:d6:8a:d6:75:9e nat
* 00:e0:4c:4f:c2:0a seven
* 00:e0:4c:47:e1:2c shark
* 00:e0:4c:46:e9:cd vic
