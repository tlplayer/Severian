# `bytes`

```sev
#Bytes are represented with B notation
a := 1B

# Bytes are used for system access to memory and seeking
import file

a := file.seek("file.txt",2B)


#This disambiguates and clarifies byte operations and memory legibility
kilo_byte = 1KB
mega_byte = 1MB
giga_byte = 1GB

#Doing comparisons/checks etc. this way avoids 1024*1024... in constants
```