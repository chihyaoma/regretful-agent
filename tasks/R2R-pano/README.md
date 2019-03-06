  
## Directory Structure
    
    .
    ├── agents          
    │   ├── pano_agent.py    # rolling out the agents and record actions
    ├── data                 # data of the Room-to-Room dataset
    ├── models               
    │   ├── encoder.py       # RNN language encoder
    │   ├── modules.py       # basic network modules
    │   ├── policy_model.py  # network architecture of the agent
    │   ├── rnn.py           # RNN with optional masking
    ├── env.py               # provide environemental changes and functions
    ├── eval.py              # evaluating agent's trajectory
    ├── main.py              # load dataset, construct agent, and start training
    ├── trainer.py           # training and evaluation iterations
    └── utils.py             # utilities functions