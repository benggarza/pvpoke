function PlayerMemory() {
    var front = null;
    var back = null;
    var length = 0;

    // Memory Node data immutable
    function MemoryNode(state, reward, action) {
        var state = state;
        var reward = reward;
        var action = action;
        var next = null;
        var prev = null;

        this.getEvent = function(){
            return {'state': state, 'reward': reward, 'action': action};
        }

        this.getNext = function(){
            return next;
        }

        this.getPrev = function(){
            return prev;
        }

        this.setNext = function(nextNode) {
            next = nextNode;
        }
        
        this.setPrev = function(prevNode) {
            prev = prevNode;
        }
    }

    this.getLength = function(){
        return length;
    }

    this.deQueue = function(){
        if (length > 0) {
            let frontNode = front;
            front = frontNode.getNext();
            frontNode.setNext(null);
            if (length == 1){
                back = null;
            } else {
                front.setPrev(null); // protected from null reference
            }
            length -= 1;
            return frontNode.getEvent();
        } else {
            return null;
        }
    }

    this.pop = function(){
        if (length > 0){
            let backNode = back;
            back = backNode.getPrev();
            backNode.setPrev(null);
            if (length == 1){
                front = null;
            } else {
                back.setNext(null);
            }
            length -= 1;
            return backNode.getEvent();
        } else {
            return null;
        }
    }

    this.addEvent = function (state, reward, action){
        let newNode = new MemoryNode(state, reward, action);
        if (length == 0) {
            front = newNode;
            back = newNode;
        } else {
            back.setNext(newNode);
            newNode.setPrev(back);
            back = newNode;
        }
        length += 1;
    }
}