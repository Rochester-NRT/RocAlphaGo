
/** 
 * This extension handles go game records(kifu). In WGo kifu is stored in JSON. Kifu structure example:
 *
 * JGO proposal = {
 *	 size: 19,
 *	 info: {
 *	 	black: {name:"Lee Chang-Ho", rank:"9p"},
 *	 	white: {name:"Lee Sedol", rank:"9p"},
 *	 	komi: 6.5,
 *   },
 *	 game: [
 *	   {B:"mm"},
 * 	   {W:"nn"},
 *	   {B:"cd"},
 *     {},
 *   ]
 * }
 *
 */
 
(function (WGo, undefined) {

"use strict";

var recursive_clone = function(node) {
	var n = new KNode(JSON.parse(JSON.stringify(node.getProperties())));
	for(var ch in node.children) {
		n.appendChild(recursive_clone(node.children[ch]));
	}
	return n;
}

var find_property = function(prop, node) {
	var res;
	if(node[prop] !== undefined) return node[prop];
	for(var ch in node.children) {
		res = find_property(prop, node.children[ch])
		if(res) return res;
	}
	return false;
}

var recursive_save = function(gameTree, node) {
	gameTree.push(JSON.parse(JSON.stringify(node.getProperties())));
	if(node.children.length > 1) {
		var nt = [];
		for(var i = 0; i < node.children.length; i++) {
			var t = [];
			recursive_save(t, node.children[i]);
			nt.push(t);
		}
		gameTree.push(nt);
	}
	else if(node.children.length) {
		recursive_save(gameTree, node.children[0]);
	}
}

var recursive_save2 = function(gameTree, node) {
	var anode = node;
	var tnode;
	
	for(var i = 1; i < gameTree.length; i++) {
		if(gameTree[i].constructor == Array) {
			for(var j = 0; j < gameTree[i].length; j++) {
				tnode = new KNode(gameTree[i][j][0]);
				anode.appendChild(tnode);
				recursive_save2(gameTree[i][j], tnode);
			}
		}
		else {
			tnode = new KNode(gameTree[i]);
			anode.insertAfter(tnode);
			anode = tnode;
		}
	}
}

/**
 * Kifu class - for storing go game record and easy manipulation with it
 */

var Kifu = function() {
	this.size = 19;
	this.info = {};
	this.root = new KNode();
	this.nodeCount = 0;
	this.propertyCount = 0;
}

Kifu.prototype ={
	constructor: Kifu,
	clone: function() {
		var clone = new Kifu();
		clone.size = this.size;
		clone.info = JSON.parse(JSON.stringify(this.info));
		clone.root = recursive_clone(this.root);
		clone.nodeCount = this.nodeCount;
		clone.propertyCount = this.propertyCount;
		return clone;
	},
	hasComments: function() {
		return !!find_property("comment", this.root);
	},
}

/**
 * Create kifu object from SGF string
 */

Kifu.fromSgf = function(sgf) {
	return WGo.SGF.parse(sgf);
}

/**
 * Create kifu object from JGO
 */

Kifu.fromJGO = function(arg) {
	var jgo = typeof arg == "string" ? JSON.parse(arg) : arg;
	var kifu = new Kifu();
	kifu.info = JSON.parse(JSON.stringify(jgo.info));
	kifu.size = jgo.size;
	kifu.nodeCount = jgo.nodeCount;
	kifu.propertyCount = jgo.propertyCount;
		
	kifu.root = new KNode(jgo.game[0]);
	recursive_save2(jgo.game, kifu.root);
	
	return kifu;
}

/**
 * Return SGF string from kifu object
 */

Kifu.prototype.toSgf = function() {
	// not implemented yet
}

/**
 * Return JGO from kifu object
 */

Kifu.prototype.toJGO = function(stringify) {
	var jgo = {};
	jgo.size = this.size;
	jgo.info = JSON.parse(JSON.stringify(this.info));
	jgo.nodeCount = this.nodeCount;
	jgo.propertyCount = this.propertyCount;
	jgo.game = [];
	recursive_save(jgo.game, this.root);
	if(stringify) return JSON.stringify(jgo);
	else return jgo;
}

var player_formatter = function(value) {
	var str;
	if(value.name) {
		str = WGo.filterHTML(value.name);
		if(value.rank) str += " ("+WGo.filterHTML(value.rank)+")";
		if(value.team) str += ", "+WGo.filterHTML(value.team);
	}
	else {
		if(value.team) str = WGo.filterHTML(value.team);
		if(value.rank) str += " ("+WGo.filterHTML(value.rank)+")";
	}
	return str;
}

/**
 * Game information formatters. Each formatter is a function which somehow formats input text.
 */

Kifu.infoFormatters = {
	black: player_formatter,
	white: player_formatter,
	TM: function(time) {
		if(time == 0) return WGo.t("none");
		
		var res, t = Math.floor(time/60);
		
		if(t == 1) res = "1 "+WGo.t("minute");
		else if(t > 1) res = t+" "+WGo.t("minutes");
		
		t = time%60;
		if(t == 1) res += " 1 "+WGo.t("second");
		else if(t > 1) res += " "+t+" "+WGo.t("seconds");
		
		return res;
	},
	RE: function(res) {
		return '<a href="javascript: void(0)" onclick="this.parentNode.innerHTML = \''+WGo.filterHTML(res)+'\'" title="'+WGo.t('res-show-tip')+'">'+WGo.t('show')+'</a>';
	},
}

/**
 * List of game information properties
 */

Kifu.infoList = ["black", "white", "AN", "CP", "DT", "EV", "GN", "GC", "ON", "OT", "RE", "RO", "RU", "SO", "TM", "PC", "KM"];

WGo.Kifu = Kifu;

var no_add = function(arr, obj, key) {
	for(var i = 0; i < arr.length; i++) {
		if(arr[i].x == obj.x && arr[i].y == obj.y) {
			arr[i][key] = obj[key];
			return;
		}
	}
	arr.push(obj);
}

var no_remove = function(arr, obj) {
	if(!arr) remove;
	for(var i = 0; i < arr.length; i++) {
		if(arr[i].x == obj.x && arr[i].y == obj.y) {
			arr.splice(i,1);
			return;
		}
	}
}

/**
 * Node class of kifu game tree. It can contain move, setup or markup properties.
 *
 * @param {object} properties
 * @param {KNode} parent (null for root node)
 */

var KNode = function(properties, parent) {
	this.parent = parent || null;
	this.children = [];
	// save all properties
	if(properties) for(var key in properties) this[key] = properties[key];
}

KNode.prototype = {
	constructor: KNode,
	
	/**
	 * Get node's children specified by index. If it doesn't exist, method returns null.
	 */
	
	getChild: function(ch) {
		var i = ch || 0;
		if(this.children[i]) return this.children[i];
		else return null;
	},
	
	/**
	 * Add setup property.
	 * 
	 * @param {object} setup object with structure: {x:<x coordinate>, y:<y coordinate>, c:<color>}
	 */
	
	addSetup: function(setup) {
		this.setup = this.setup || [];
		no_add(this.setup, setup, "c");
		return this;
	},
	
	/**
	 * Remove setup property.
	 * 
	 * @param {object} setup object with structure: {x:<x coordinate>, y:<y coordinate>}
	 */
	
	removeSetup: function(setup) {
		no_remove(this.setup, setup);
		return this;
	},
	
	/**
	 * Add markup property.
	 * 
	 * @param {object} markup object with structure: {x:<x coordinate>, y:<y coordinate>, type:<type>}
	 */
	
	addMarkup: function(markup) {
		this.markup = this.markup || [];
		no_add(this.markup, markup, "type");
		return this;
	},
	
	/**
	 * Remove markup property.
	 * 
	 * @param {object} markup object with structure: {x:<x coordinate>, y:<y coordinate>}
	 */
	
	removeMarkup: function(markup) {
		no_remove(this.markup, markup);
		return this;
	},
	
	/**
	 * Remove this node.
	 * Node is removed from its parent and children are passed to parent.
	 */
	
	remove: function() {
		var p = this.parent;
		if(!p) throw new Exception("Root node cannot be removed");
		for(var i in p.children) {
			if(p.children[i] == this) {
				p.children.splice(i,1);
				break;
			}
		}
		p.children = p.children.concat(this.children);
		this.parent = null;
		return p;
	},
	
	/**
	 * Insert node after this node. All children are passed to new node.
	 */
	
	insertAfter: function(node) {
		node.children = node.children.concat(this.children);
		node.parent = this;
		this.children = [node];
		return node;
	},
	
	/**
	 * Append child node to this node.
	 */
	
	appendChild: function(node) {
		node.parent = this;
		this.children.push(node);
		return node;
	},
	
	/**
	 * Get properties as object.
	 */
	
	getProperties: function() {
		var props = {};
		for(var key in this) {
			if(this.hasOwnProperty(key) && key != "children" && key != "parent") props[key] = this[key];
		}
		return props;
	}
}

WGo.KNode = KNode;

var pos_diff = function(old_p, new_p) {
	var size = old_p.size, add = [], remove = [];
	
	for(var i = 0; i < size*size; i++) {
		if(old_p.schema[i] && !new_p.schema[i]) remove.push({x:Math.floor(i/size),y:i%size});
		else if(old_p.schema[i] != new_p.schema[i]) add.push({x:Math.floor(i/size),y:i%size,c:new_p.schema[i]});
	}
	
	return {
		add: add,
		remove: remove
	}
}

/**
 * KifuReader object is capable of reading a kifu nodes and executing them. It contains Game object with actual position.
 * Variable change contains last changes of position.
 * If parameter rememberPath is set, KifuReader will remember last selected child of all nodes.
 */

var KifuReader = function(kifu, rememberPath) {
	this.kifu = kifu;
	this.node = this.kifu.root;
	this.game = new WGo.Game(this.kifu.size);
	this.path = {m:0};

	this.change = exec_node(this.game, this.node, true);
	if(this.kifu.info["HA"] && this.kifu.info["HA"] > 1) this.game.turn = WGo.W;
	
	if(rememberPath) this.rememberPath = true;
	else this.rememberPath = false;
}

var set_subtract = function(a, b) {
	var n = [], q;
	for(var i in a) {
		q = true;
		for(var j in b) {
			if(a[i].x == b[j].x && a[i].y == b[j].y) {
				q = false;
				break;
			}
		}
		if(q) n.push(a[i]);
	}
	return n;
}

var concat_changes = function(ch_orig, ch_new) {
	ch_orig.add = set_subtract(ch_orig.add, ch_new.remove).concat(ch_new.add);
	ch_orig.remove = set_subtract(ch_orig.remove, ch_new.add).concat(ch_new.remove);
}

// change game object according to node, return changes
var exec_node = function(game, node, first) {
	if(node.parent) node.parent._last_selected = node.parent.children.indexOf(node);
	
	if(node.move != undefined) {
		if(node.move.pass) {
			game.pass(node.move.c);
			return {add:[], remove:[]};
		}
		else {
			var res = game.play(node.move.x, node.move.y, node.move.c);
			if(typeof res == "number") throw new InvalidMoveError(res, node);
			return {
				add: [node.move],
				remove: res
			}
		}
	}
	else if(node.setup != undefined) {
		if(!first) game.pushPosition();
		
		var add = [], remove = [];
		
		for(var i in node.setup) {
			if(node.setup[i].c) {
				game.addStone(node.setup[i].x, node.setup[i].y, node.setup[i].c);
				add.push(node.setup[i]);
			}
			else {
				game.removeStone(node.setup[i].x, node.setup[i].y);
				remove.push(node.setup[i]);
			}
		}
		
		// TODO: check & test handling of turns 
		if(node.turn) game.turn = node.turn;
		
		return {
			add: add,
			remove: remove
		};
	}
	else if(!first) {
		game.pushPosition();
	}
	return {add:[], remove:[]};
}

var exec_next = function(i) {
	if(i === undefined && this.rememberPath) i = this.node._last_selected;
	i = i || 0;
	var node = this.node.children[i];
	
	if(!node) return false;
	
	var ch = exec_node(this.game, node);
	
	this.path.m++;
	if(this.node.children.length > 1) this.path[this.path.m] = i;
	
	this.node = node;
	return ch;
}

var exec_previous = function() {
	if(!this.node.parent) return false;
	
	this.node = this.node.parent;
	
	this.game.popPosition();
	
	this.path.m--;
	if(this.path[this.path.m] !== undefined) delete this.path[this.path.m];
	
	return true;
}

var exec_first = function() {
	//if(!this.node.parent) return;
	
	this.game.firstPosition();
	this.node = this.kifu.root;
	
	this.path = {m: 0};
	
	this.change = exec_node(this.game, this.node, true);
	if(this.kifu.info["HA"] && this.kifu.info["HA"] > 1) this.game.turn = WGo.W;
}

KifuReader.prototype = {
	constructor: KifuReader,
	
	/**
	 * Go to next node and if there is a move play it.
	 */
	
	next: function(i) {
		this.change = exec_next.call(this, i);
		return this;
	},
	
	/**
	 * Execute all nodes till the end.
	 */
	
	last: function() {
		var ch;
		this.change = {
			add: [],
			remove: []
		}
		while(ch = exec_next.call(this)) concat_changes(this.change, ch);
		return this;
	},
	
	/**
	 * Return to the previous position (redo actual node) 
	 */
	
	previous: function() {	
		var old_pos = this.game.getPosition();
		exec_previous.call(this);
		this.change = pos_diff(old_pos, this.game.getPosition());
		return this;
	},
	
	/**
	 * Go to the initial position
	 */
	
	first: function() {
		var old_pos = this.game.getPosition();
		exec_first.call(this);		
		this.change = pos_diff(old_pos, this.game.getPosition());
		return this;
	},
	
	/**
	 * Go to position specified by path object
	 */
	
	goTo: function(path) {
		if(path === undefined) return this;
		
		var old_pos = this.game.getPosition();

		exec_first.call(this);
		
		var r;
		
		for(var i = 0; i < path.m; i++) {
			if(!exec_next.call(this, path[i+1])) {
				break;
			}
		}
		
		this.change = pos_diff(old_pos, this.game.getPosition());
		return this;
	},
	
	/**
	 * Go to previous fork (a node with more than one child)
	 */
	
	previousFork: function() {
		var old_pos = this.game.getPosition();
		while(exec_previous.call(this) && this.node.children.length == 1);
		this.change = pos_diff(old_pos, this.game.getPosition());
		return this;
	},
	
	/**
	 * Shortcut. Get actual position object.
	 */
	
	getPosition: function() {
		return this.game.getPosition();
	}
}

WGo.KifuReader = KifuReader;

// Class handling invalid moves in kifu
var InvalidMoveError = function(code, node) {
	this.name = "InvalidMoveError";
    this.message = "Invalid move in kifu detected. ";
	
	if(node.move && node.move.c !== undefined && node.move.x !== undefined && node.move.y !== undefined) this.message += "Trying to play "+(node.move.c == WGo.WHITE ? "white" : "black")+" move on "+String.fromCharCode(node.move.x+65)+""+(19-node.move.y);
	else this.message += "Move object doesn't contain arbitrary attributes.";
	
	if(code) {
		switch(code) {
			case 1:
				this.message += ", but these coordinates are not on board.";
			break;
			case 2:
				this.message += ", but there already is a stone.";
			break;
			case 3:
				this.message += ", but this move is a suicide.";
			case 4:
				this.message += ", but this position already occured.";
			break;
		}
	}
	else this.message += "."
}
InvalidMoveError.prototype = new Error();
InvalidMoveError.prototype.constructor = InvalidMoveError;

WGo.InvalidMoveError = InvalidMoveError;

WGo.i18n.en["show"] = "show";
WGo.i18n.en["res-show-tip"] =  "Click to show result.";

})(WGo);
