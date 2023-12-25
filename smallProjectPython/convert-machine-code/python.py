import pathlib
Opcode ={
            "add":"0110011",
            "sub":"0110011",
            "sll":"0110011",
            "slt":"0110011",
            "sltu":"0110011",
            "xor":"0110011",
            "srl":"0110011",
            "sra":"0110011",
            "or":"0110011",
            "and":"0110011",
            "addw":"0110011",
            "subw":"0110011",
            "srlw":"0110011",
            "sllw":"0110011",
            "sraw":"0110011",
            "lui":"0110111",
            "beq":"1100011",
            "bne":"1100011",
            "blt":"1100011",
            "bge":"1100011",
            "bltu":"1100011",
            "bgeu":"1100011",
            "jalr":"1100111",
            "jal":"1101111",
            "lb":"0000011",
            "lh":"0000011",
            "lw":"0000011",
            "ld":"0000011",
            "lbu":"0000011",
            "lwu":"0000011",
            "addi":"0010011",
            "slli":"0010011",
            "slti":"0010011",
            "sltiu":"0010011",
            "xori":"0010011",
            "srli":"0010011",
            "srai":"0010011",
            "ori":"0010011",
            "andi":"0010011",
            "auipc":"0010111",
            "addiw":"0011011",
            "slliw":"0011011",
            "srliw":"0011011",
            "sraiw":"0011011",
            "sb":"0100011",
            "sh":"0100011",
            "sw":"0100011",
            "sd":"0100011",
            "lhu":"0000011"
            }
Function3={
            "add":"000",
            "sub":"000",
            "sll":"001",
            "slt":"010",
            "sltu":"011",
            "xor":"100",
            "srl":"101",
            "sra":"101",
            "or":"110",
            "and":"111",
            "addw":"000",
            "subw":"000",
            "sllw":"001",
            "srlw":"101",
            "sraw":"101",
            "beq":"000",
            "bne":"001",
            "blt":"100",
            "bge":"101",
            "bltu":"110",
            "bgeu":"111",
            "jalr":"000",
            "lb":"000",
            "lw":"010",
            "ld":"011",
            "lbu":"100",
            "lwu":"110",
            "addi":"000",
            "slli":"001",
            "slti":"010",
            "sltiu":"011",
            "xori":"100",
            "srli":"101",
            "srai":"101",
            "ori":"110",
            "andi":"111",
            "addiw":"000",
            "slliw":"001",
            "srliw":"101",
            "sraiw":"101",
            "sb":"000",
            "sw":"010",
            "sd":"011",
            "lh":"001",
            "sh":"001",
            "lhu":"101"
            }
Function7={
            "add":"0000000",
            "sub":"0100000",
            "sll":"0000000",
            "slt":"0000000",
            "sltu":"0000000",
            "xor":"0000000",
            "srl":"0000000",
            "sra":"0100000",
            "or":"0000000",
            "and":"0000000",
            "slli":"0000000",
            "srli":"0000000",
            "srai":"0100000",
            "slliw":"0000000",
            "srliw":"0000000",
            "sraiw":"0100000"
            }
Label={
            }
Type ={
            "R":["add","sub","sll","srl","slt","sltu","xor","sra","or","and","andw","subw","sllw","srlw","sraw"],
            "I":["lb","lw","ld","lbu","lwu","addi","slli","slti","sltiu","xori","srli","srai","ori","andi","addiw","slliw","srliw","sraiw","jalr","sd","lh"],
            "SB":["beq","bne","blt","bge","bltu","bgeu"],
            "U":["auipc","lui"],
            "UJ":["jal"],
            "S":["sb","sw","sh"]
            }
      
def ToBinary(decimal,length):
    binary=""
    decimal=int(decimal)
    if(decimal>=0):
        if(decimal==0):
            binary+=str(decimal)
            return binary.rjust(length,"0")
        while(decimal!=0):
           binary+=str(decimal%2)
           decimal=decimal//2
        binary=binary[::-1]
        return binary.rjust(length,"0")
    else:
        binary = bin(decimal & int("1"*length, 2))[2:]
        return binary
def DeleteBlank(data):
   for i in range(len(data)):
       data[i]=data[i].strip()
   return data
def FindLabel(data):
   for i in range(len(data)):
     mark=data[i].find(":")
     if(mark!=-1):
        label=data[i][:mark]
        Label[label]=i 
        data[i]=data[i].replace(data[i],"")
   for x in range(len(Label)):
     data.remove("")
def FindType(type):
   for value,key in Type.items():
       for i in key:
           if(i==type):
              return value
def GenerateBin(data,result):
    result=[]
    for i in range(len(data)):
                binary_code=""
                cmd=""
                number=""
                for index in range(len(data[i])):
                     if(data[i][index]==" "):
                            break
                     cmd+=data[i][index]
                cmd=cmd.strip()
                type=FindType(cmd)
                if(data[i]!=""):
                    if(type=="R"):
                        for value,key in Function7.items():               
                            if(cmd==value):
                                binary_code+=str(key)
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        number=data[i].split(',')
                        for i in range(len(number)):
                            number[i]=number[i].strip()
                            number[i]=number[i].replace("x","")
                            number[i]=number[i].replace("(","")
                            number[i]=number[i].replace(")","")
                        number=number[::-1]
                        for i in range(0,len(number)-1):
                            binary_code+=ToBinary(int(number[i]),5)
                        for value,key in Function3.items():
                            if(cmd==value):
                                binary_code+=str(key)
                        binary_code+=ToBinary(int(number[2]),5)
                        for value,key in Opcode.items():
                            if(cmd==value):
                                binary_code+=str(key)
                    elif(type=="I"):
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        if(cmd=="lw" or cmd == "ld" or cmd == "lb" or cmd=="lbu" or cmd=="lwu" or cmd=="lh" or cmd=="lhu"):
                            number=data[i].replace("(",",")
                            number=number.split(',')
                            for i in range(len(number)):
                                number[i]=number[i].strip()
                                number[i]=number[i].replace("x","")
                                number[i]=number[i].replace(")","")
                            binary_code+=ToBinary(int(number[1]),12)
                            binary_code+=ToBinary(int(number[2]),5)
                            for value,key in Function3.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                            binary_code+=ToBinary(int(number[0]),5)
                            for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                        elif(cmd=="srai" or cmd == "slli" or cmd == "srli"):
                            for value,key in Function7.items():               
                                if(cmd==value):
                                    binary_code+=str(key)
                            number=data[i].split(',')
                            for i in range(len(number)):
                                number[i]=number[i].strip()
                                number[i]=number[i].replace("x","")
                                number[i]=number[i].replace(")","")
                                number[i]=number[i].replace("(","")
                            binary_code+=ToBinary(int(number[2]),5)
                            binary_code+=ToBinary(int(number[1]),5)
                            for value,key in Function3.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                            binary_code+=ToBinary(int(number[0]),5)
                            for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                        else:
                            number=data[i].split(',')
                            for i in range(len(number)):
                                number[i]=number[i].strip()
                                number[i]=number[i].replace("x","")
                                number[i]=number[i].replace("(","")
                                number[i]=number[i].replace(")","")
                                number[i]=number[i].replace(",","")
                            binary_code+=ToBinary(int(number[2]),12)
                            binary_code+=ToBinary(int(number[1]),5)
                            for value,key in Function3.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                            binary_code+=ToBinary(int(number[0]),5)
                            for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                    elif(type=="U"):
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        number=data[i].split(',')
                        for i in range(len(number)):
                            number[i]=number[i].strip()
                            number[i]=number[i].replace("x","")
                            number[i]=number[i].replace("(","")
                            number[i]=number[i].replace(")","")
                        binary_code+=ToBinary(int(number[1]),20)
                        binary_code+=ToBinary(int(number[0]),5)
                        for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                    elif(type=="UJ"):
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        number=data[i].split(',')
                        for i in range(len(number)):
                            number[i]=number[i].strip()
                            number[i]=number[i].replace("x","")
                            number[i]=number[i].replace("(","")
                            number[i]=number[i].replace(")","")
                        name_label=number[1]
                        for value,key in Label.items():
                            if(name_label==value):
                                distance=key-i-1
                        binary_code+=ToBinary(distance,20)
                        binary_code+=ToBinary(int(number[0]),5)
                        for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)           
                    elif(type=="SB"):
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        number=data[i].split(',')
                        for i in range(len(number)):
                            number[i]=number[i].strip()
                            number[i]=number[i].replace("x","")
                            number[i]=number[i].replace("(","")
                            number[i]=number[i].replace(")","")
                        name_label=number[2]
                        for value,key in Label.items():
                            if(name_label==value):
                                distance=key-i-1
                        distance= ToBinary(int(distance),12)
                        binary_code+=str(distance[:7])
                        binary_code+=ToBinary(int(number[0]),5)
                        binary_code+=ToBinary(int(number[1]),5)
                        for value,key in Function3.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                        binary_code+=str(distance[7:])
                        for value,key in Opcode.items():
                                if(cmd==value):
                                    binary_code+=str(key)
                    else:   
                        data[i]=data[i].replace(cmd,"")
                        data[i]=data[i].lstrip()
                        number=data[i].replace("(",",")
                        number=number.split(',')
                        for i in range(len(number)):
                            number[i]=number[i].strip()
                            number[i]=number[i].replace("x","")
                            number[i]=number[i].replace(")","")
                        offset=ToBinary(int(number[1]),12)
                        binary_code+=str(offset[:7])
                        binary_code+=ToBinary(int(number[0]),5)
                        for value,key in Function3.items():
                            if(cmd==value):
                                binary_code+=str(key)
                        binary_code+=str(offset[7:])
                        binary_code+=ToBinary(int(number[2]),5)
                        for value,key in Opcode.items():
                            if(cmd==value):
                                binary_code+=str(key)
                else: continue
                result.append(binary_code)
    return result
def Convert(path,position_text,path_write):
    for name in path.glob('**/*.S'):
        file=open(name)
        data=file.readlines()
        print(f">>> {name}")
        for i in range(len(data)):
            mark=data[i].find("#")
            mark1=data[i].find(".")
            mark2=data[i].find("//")
            if(mark==0 or mark1==0 or mark2 ==0):
                data[i]=data[i][:-len(data[i])]
            elif(mark!=-1):
                data[i]=data[i][:mark-1]
            elif(mark1!=-1):
                data[i]=data[i][:mark1-1]
            elif(mark2!=-1):
                data[i]=data[i][:mark2-1]
        result=[]
        FindLabel(data)
        DeleteBlank(data)
        result=GenerateBin(data,result)
        result="\n".join(result)
        writing=open(f'{path_write}/Machine code {position_text}.txt',"w")
        writing.writelines(result)
        writing.close()
        print(f">>> {path_write}/Machine code {position_text}.txt\n")
        position_text+=1
        file.close()
if __name__=="__main__":
    position_text=0
    path_1 = pathlib.Path('Directory/assembly_1')
    path_write_1="Directory/machine code/assembly_1"
    path_2 = pathlib.Path('Directory/assembly_2')
    path_write_2="Directory/machine code/assembly_2"
    Convert(path_1,position_text,path_write_1) 
    Convert(path_2,position_text,path_write_2)
