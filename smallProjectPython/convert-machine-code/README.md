# convert-machine-code
In this project i will convert all instruction in RICSV to machine code. <br /> 
All instruction of RICSV will in folder **Directory** and each file will have some comments or labels, spaces,... so my mission is delete all comments or labels,... then covert it. <br />
All steps of project: <br />
  1. Final the label because it can be effected by instruction of Type "J",....
  2. Delete blank (because the raw data will contains blank or space)
  3. Then we find Type of instruction, each instruction has the type of its and with that types we can find the binary code through the references table
  4. Next we write the binary code in folder **machine code**
