
(function(WGo, undefined){

WGo.SGF = {};

var to_num = function(str, i) {
	return str.charCodeAt(i)-97;
}

var sgf_player_info = function(type, black, kifu, node, value, ident) {
	var c = ident == black ? "black" : "white";
	kifu.info[c] = kifu.info[c] || {};
	kifu.info[c][type] = value[0];
}

// handling properties specifically
var properties = WGo.SGF.properties = {}

// Move properties
properties["B"] = properties["W"] = function(kifu, node, value, ident) {
	if(!value[0] || (kifu.size <= 19 && value[0] == "tt")) node.move = {
		pass: true,
		c: ident == "B" ? WGo.B : WGo.W
	};
	else node.move = {
		x: to_num(value[0], 0), 
		y: to_num(value[0], 1), 
		c: ident == "B" ? WGo.B : WGo.W
	};
}
	
// Setup properties
properties["AB"] = properties["AW"] = function(kifu, node, value, ident) {
	for(var i in value) {
		node.addSetup({
			x: to_num(value[i], 0), 
			y: to_num(value[i], 1), 
			c: ident == "AB" ? WGo.B : WGo.W
		});
	}
}
properties["AE"] = function(kifu, node, value) {
	for(var i in value) {
		node.addSetup({
			x: to_num(value[i], 0), 
			y: to_num(value[i], 1), 
		});
	}
}
properties["PL"] = function(kifu, node, value) {
	node.turn = value[0] == "b" ? WGo.B : WGo.W;
}
	
// Node annotation properties
properties["C"] = function(kifu, node, value) {
	node.comment = value.join();
}
	
// Markup properties
properties["LB"] = function(kifu, node, value) {
	for(var i in value) {
		node.addMarkup({
			x: to_num(value[i],0), 
			y: to_num(value[i],1), 
			type: "LB", 
			text: value[i].substr(3)
		});
	}
}
properties["CR"] = properties["SQ"] = properties["TR"] = properties["SL"] = properties["MA"] = function(kifu, node, value, ident) {
	for(var i in value) {
		node.addMarkup({
			x: to_num(value[i],0), 
			y: to_num(value[i],1), 
			type: ident
		});
	}
}

// Root properties
properties["SZ"] = function(kifu, node, value) {
	kifu.size = parseInt(value[0]);
}
	
// Game info properties
properties["BR"] = properties["WR"] = sgf_player_info.bind(this, "rank", "BR");
properties["PB"] = properties["PW"] = sgf_player_info.bind(this, "name", "PB");
properties["BT"] = properties["WT"] = sgf_player_info.bind(this, "team", "BT");
properties["TM"] =  function(kifu, node, value, ident) {
	kifu.info[ident] = value[0];
	node.BL = value[0];
	node.WL = value[0];
}

var reg_seq = /\(|\)|(;(\s*[A-Z]+\s*((\[\])|(\[(.|\s)*?([^\\]\])))+)*)/g;
var reg_node = /[A-Z]+\s*((\[\])|(\[(.|\s)*?([^\\]\])))+/g;
var reg_ident = /[A-Z]+/;
var reg_props = /(\[\])|(\[(.|\s)*?([^\\]\]))/g;

// parse SGF string, return WGo.Kifu object
WGo.SGF.parse = function(str) { 

	var stack = [],
		sequence, props, vals, ident,
		kifu = new WGo.Kifu(),
		node = null;
		
	// make sequence of elements and process it
	sequence = str.match(reg_seq);
	
	for(var i in sequence) {
		// push stack, if new variant
		if(sequence[i] == "(") stack.push(node);
		
		// pop stack at the end of variant
		else if(sequence[i] == ")") node = stack.pop();
		
		// reading node (string starting with ';')
		else {
			// create node or use root
			if(node) kifu.nodeCount++;
			node = node ? node.appendChild(new WGo.KNode()) : kifu.root;
			
			// make array of properties
			props = sequence[i].match(reg_node);
			kifu.propertyCount += props.length;
			
			// insert all properties to node
			for(var j in props) {
				// get property's identificator
				ident = reg_ident.exec(props[j])[0];
				
				// separate property's values
				vals = props[j].match(reg_props);
				
				// remove additional braces [ and ]
				for(var k in vals) vals[k] = vals[k].substring(1, vals[k].length-1).replace(/\\(?!\\)/g, "");
				
				// call property handler if any
				if(WGo.SGF.properties[ident]) WGo.SGF.properties[ident](kifu, node, vals, ident);
				else {
					// if there is only one property, strip array
					if(vals.length <= 1) vals = vals[0];
					
					// default node property saving
					if(node.parent) node[ident] = vals;
					
					// default root property saving
					else {
						kifu.info[ident] = vals;
					}
				}
			}
		}
	}
	
	return kifu;		
}
})(WGo);