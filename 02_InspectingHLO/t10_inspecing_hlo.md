This tutorial shows you the output of code generation:

- Set the following environment variable in your terminal window (Unix/MacOS style), when running a program this will generation output debug files
    export XLA_FLAGS=--xla_dump_to=NewDirectory

- Run the previous tutorial 01_...<br>
  Under NewDirectory you can now see a number of files starting with name:<br>
    module_0000.UniqueNameHere.number.<br>
  UniqueNameHere is the computation name we provided in t05 as argument to XlaBuilder

- These files are generated as part of the compilation process to describe the
  description of our program, in protobuf format, and also how this changes as the 
  the compilation passes are executing

- Let's look at the files in detail, in the order in which they are generated by XLA,
  omitting the initial NewDirectory/module_0000.UniqueNameHere.number. prefix

1. before_optimizations.txt


2. cpu_after_optimizations.txt


3. cpu_after_optimizations-buffer-assignment.txt


4. ir-no-opt.ll


5. ir-no-opt-noconst.ll


6. ir-with-opt.ll


7. ir-with-opt-noconst.ll


8. o

