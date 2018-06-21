This README is intended to help the execution of agents in a slurm client (I used OSIRIM).

The directory mario-ai-master_OSIRIM should be already configured to be used on OSIRIM.

After configuring the agent in mario-ai-master_OSIRIM you need to copy this directory to your slurm workspace, you can use:

```scp -r mario-ai-master_OSIRIM YOURLOGIN@osirim-slurm.irit.fr:.```

Replace YOURLOGIN with your login to the slurm client. You can copy the directory in a specific path with:

```... YOURLOGIN@osirim-slurm.irit.fr:./my/specific/path```

Connect to OSIRIM:

```ssh osirim-slurm.irit.fr -l YOURLOGIN```

Go to the directory then run:

```sbatch runDQN.sh```

If it doesn't works maybe you'll need to modify Makefile.Linux and ch_idsia_tools_amico_AmiCoJavaPy.cc like it's explained in README.md.
You also need to have no graphic displayed, to disable the graphic display of MarioAI Benchmark you need to use `marioAIOptions.setVisualization(false)` in src/main/java/ch/idsia/scenarios/Main.java.

You can show a report for your running (or pending) jobs with:

```squeue -u YOURLOGIN```

To know more about slurm options I recommand the Tutoriel-Slurm-V4-2.pdf file in this directory.
