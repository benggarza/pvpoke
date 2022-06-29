//import * as tf from '@tensorflow/tfjs';

function PlayerModel(b, hiddenLayerSizesOrModel, numStates, numActions, batchSize) {
    var numStates = numStates;
    var numActions = numActions;
    var batchSize = batchSize;

    const optimizer = 'sgd';
    const loss = 'meanSquaredError';

    var network = null;

    var Q ={};
    var alpha = 0.06;
    var eps = 1.0;
    var gamma = 0.1;

    var memory = new PlayerMemory();

    this.defineModel = async function(hiddenLayerSizes) {

        // check if previous model exists on server
        // there must be a cleaner way to do this
        var xhr = new XMLHttpRequest();
        xhr.open('HEAD', "http://localhost"+webRoot+"data/training/ainetwork/model.json", false);
        xhr.send();
        console.log(xhr.status);

        if (xhr.status == 200) {
            network = await tf.loadLayersModel("http://localhost"+webRoot+"data/training/ainetwork/model.json");
            console.log(typeof network);
        } else {
            if (!Array.isArray(hiddenLayerSizes)){
                hiddenLayerSizes = [hiddenLayerSizes];
            }

            network = tf.sequential();
            hiddenLayerSizes.forEach((hiddenLayerSize, i) => {
                network.add(tf.layers.dense({
                    units: hiddenLayerSize,
                    activation: 'relu',
                    inputShape: ((i == 0) ? [numStates] : undefined),
                    kernelInitializer: 'heUniform',
                    biasInitializer: 'heUniform'
                }));
            });
            network.add(tf.layers.dense({units: numActions, activation: 'linear'}));
        }

        network.summary();
        network.compile({optimizer: optimizer, loss: loss, metrics: ['acc']});
    }

    // format state dictionary into array appropriate for network model
    this.formatState = function(state){
        // enforce order by sorting keys
        sortedKeys = Object.keys(state).sort();
        formattedState = [];
        sortedKeys.forEach(key => formattedState.push(state[key]));
        return formattedState;
    }

    // format policy dictionary into array appropriate for network model
    this.formatPolicy = function(policy){
        return [policy['fast'], policy['charged1'], policy['charged2'],policy['switch1'], policy['switch2']];
    }

    this.predict = function(state) {
        // format state
        let stateArr = tf.stack([this.formatState(state)]);
        let predictionsArr = tf.tidy(() => network.predict(stateArr)).arraySync();
        return predictionsArr[0];
    }

    this.train = async function(xBatch, yBatch) {
        let formattedXBatch = [];
        let formattedYBatch = [];
        // format xBatch
        for (const state of xBatch) {
            formattedXBatch.push(this.formatState(state));
        }

        // format yBatch
        for (const policy of yBatch) {
            formattedYBatch.push(this.formatPolicy(policy));
        }

        // can make a metrics function here to refactor later
        //console.log("x");
        //console.log(formattedXBatch);
        //console.log("YTrue");
        //console.log(formattedYBatch);

        console.log("Fitting network to new data");
        const h = await network.fit(tf.stack(formattedXBatch), tf.stack(formattedYBatch));
        console.log("Network accuracy: " + h.history.acc);
        console.log("Network loss: " + h.history.loss);
    }

    // action mapping:
    // 0: fast move
    // 1: charged move 1
    // 2: charged move 2
    // 3: switch pokemon 1
    // 4: switch pokemon 2
    this.chooseAction = function(state) {
        let action = this.bestAction(state);
        // randomness inserted here
        if (Math.random() < eps) {
            // range -2 to 2
            let actionNum = Math.floor(Math.random() * numActions);
            switch (actionNum) {
                case 0: 
                    action = 'fast';
                    break;
                case 1: 
                    action = 'charged1';
                    break;
                case 2: 
                    action = 'charged2';
                    break;
                case 3:
                    action = 'switch1';
                    break;
                case 4:
                    action = 'switch2';
                    break;
            }
            console.log("Randomly choosing action " + action);
        }  

        return action;
    }

    // returns the expected rewards for each action given a state and Q-function
    // initializes table values if not previously existed
    this.policy = function(state) {
        stateKey = this.formatState(state);
        if (!(stateKey in Q)) {
            Qvalues = this.predict(state);
            Q[stateKey] = {
                'fast': Qvalues[0],
                'charged1': Qvalues[1],
                'charged2': Qvalues[2],
                'switch1': Qvalues[3],
                'switch2': Qvalues[4]
            };
        }
        if (Math.random() < 0.005){
            console.log(Q[stateKey]);
        }
        return Q[stateKey];
    };

    // returns the expected reward for a specific state-action and Q-function
    this.eReward = function(state, action) {
        return this.policy(state)[action];
    }

    this.bestAction = function(state) {
        Qvalues = this.policy(state);

        // default to fast move, it makes sense i guess
        let bestReward = Qvalues['fast'];
        let best = 'fast';
        for (const action of Object.keys(Qvalues)){
            if (Qvalues[action] > bestReward) {
                best = action;
                bestReward = Qvalues[action];
            }
        }
        return best;
    }

    this.addEvent = function(state, reward, action) {
        memory.addEvent(state, reward ,action);
    }

    this.updateQ = function(state, action, reward, newState) {
        let policy = this.policy(state);
        let eFutureReward = this.policy(newState)[this.bestAction(newState)];
        // lines are separated to ensure that Q Table for current state is initialized via eReward()
        let eRewardChange = reward + gamma*eFutureReward - policy[action];
        let oldQ = policy[action];
        policy[action] += alpha*(eRewardChange);

        let formattedState = this.formatState(state);
        Q[formattedState] = policy;
        if (Math.random() < 0.01){
            console.log("Before update, Q value is ");
            console.log(oldQ);
            console.log("After updating "+action+" by "+alpha*eRewardChange+ ", policy is ");
            console.log(policy[action]);
            console.log(Q[formattedState]);
        }
    }

    this.update = async function(){
        //console.log("Q Table currently looks like this: ");
        //console.log(Q);
        modelXBatch = [];
        modelYBatch = [];
        let event = null;
        let prevEvent = memory.pop();
        console.log(memory.getLength() + " events in memory leading to " + memory.getLength()*256 + " Q table updates");
        while(memory.getLength() > 0) {
            // Update Q Tables
            event = prevEvent;
            //event = memory.deQueue();
            prevEvent = memory.pop(); // work backwards for more accurate Q updates

            curState = prevEvent['state'];
            action = prevEvent['action'];
            reward = event['reward'];
            newState = event['state'];

            if (action !== null) {

                // don't add trivial events for network to train on
                // otherwise network is encouraged to always choose fast moves
                curStateDupes = this.duplicateState(curState);
                //newStateDupes = this.duplicateState(newState);
                let dupeAction;
                for (let b = 0; b < 256; b++){
                    // if poke used a charged attack this turn, then the action needs to be changed for the swapped state duplicates
                    if ((action == 'charged1') && (b & 1)){
                        //console.log('changing network action to charged2 for swapped state');
                        dupeAction = 'charged2';
                    }
                    else if ((action == 'charged2') && (b & 1)){
                        //console.log('changing network action to charged1 for swapped state');
                        dupeAction = 'charged1';
                    }
                    else if ((action == 'switch1') && (b & 2)){
                        //console.log('changing network action to switch2 for swapped state');
                        dupeAction = 'switch2';
                    }
                    else if ((action == 'switch2') && (b & 2)){
                        //console.log('changing network action to switch1 for swapped state');
                        dupeAction = 'switch1';
                    }
                    else{
                        dupeAction = action;
                    }

                    // all permutations of the new state are valid so lets update the q table for all of them
                    // this line is too much to handle lol
                    // shouldn't make a difference in the end
                    /*newStateDupes.forEach(newStateDupe => {
                        this.updateQ(curStateDupes[b], dupeAction, reward, newStateDupe);
                    });*/
                    this.updateQ(curStateDupes[b], dupeAction, reward, newState);

                    modelXBatch.push(curStateDupes[b]);
                }
            }
        }

        //console.log("After updates, Q table looks like this:");
        //console.log(Q);

        console.log("Training network on " + modelXBatch.length + "events");

        // build training y set in separate loop to ensure Q tables are fully updated
        for (const state of modelXBatch) {
            modelYBatch.push(this.policy(state));
        }

        // Run network training
        this.train(modelXBatch, modelYBatch);

        // save updated results to server

        console.log("saving and loading to localhost"+webRoot+"data/network.php");
        const saveResult = await network.save("http://localhost"+webRoot+"data/network.php");
        console.log(saveResult);

    }

	this.swapStateValues = function(state, re, prefix){
		let stateKeys = Object.keys(state);
		let keyMatches = stateKeys.map(key => key.match(re)).filter(key => key);
		keyMatches.forEach(match => {let keyA = prefix + '1' + match[1];
										let keyB = prefix + '2' + match[1];
										let temp = state[keyA];
										state[keyA] = state[keyB];
										state[keyB] = temp;});
	}

	this.duplicateState = function(state){
		//TODO duplicate state with all combinations of party pokemon orders, charged move orders, opponent party pokemon orders, opponent charged move orders
		// increases data by 256 times
		let states = [];
		let stateKeys = Object.keys(state);

		let pokeChargedre = new RegExp('^p\.charged1(\..*)$');
		let partyre = new RegExp('^party\.1(\..*)$');
		let partyChargedre = new RegExp('^party\.1\.charged1(\..*)$');
		let oppChargedre = new RegExp('^o.charged1(\..*)$');
		let opartyre = new RegExp('^O.party.1(\..*)$');
		let opartyChargedre = new RegExp('^O.party.1\.charged1(\..*)$')


		// for each permutation of...
		// assigns each swap option a bit for a total of 8 bits
		for (let bitOptions = 0; bitOptions < 256; bitOptions++){
			let stateDup = {}
			Object.assign(stateDup, state);
			// lead charged moves
			if (bitOptions  & 1){
				this.swapStateValues(stateDup, pokeChargedre, 'p.charged');
			}
			// party pokemon
			if (bitOptions & 2){
				this.swapStateValues(stateDup, partyre, 'party.')
			}
			// party pokemon 1 charged moves
			if (bitOptions & 4){
				this.swapStateValues(stateDup, partyChargedre, 'party.1.charged');
			}
			// party pokemon 2 charged moves
			if (bitOptions & 8){
				this.swapStateValues(stateDup, partyChargedre, 'party.2.charged');
			}
			// opponent lead charged moves
			if (bitOptions & 16){
				this.swapStateValues(stateDup, oppChargedre, 'o.charged');
			}
			// opponent party pokemon
			if (bitOptions & 32){
				this.swapStateValues(stateDup, opartyre, 'O.party.');
			}
			// opponent party pokemon 1 charged moves
			if (bitOptions & 64){
				this.swapStateValues(stateDup, opartyChargedre, 'O.party.1.charged');
			}
			// opponent party pokemon 2 charged moves
			if (bitOptions & 128){
				this.swapStateValues(stateDup, opartyChargedre, 'O.party.2.charged');
			}
			states.push(stateDup);
		}

		return states;
	}
}