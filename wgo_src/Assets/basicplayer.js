
(function(WGo){

"use strict";

// player counter - for creating unique ids
var pl_count = 0;

// generate DOM of region
var playerBlock = function(name, parent, visible) {
	var e = {};
	e.element = document.createElement("div");
	e.element.className = "wgo-player-"+name;
	e.wrapper = document.createElement("div");
	e.wrapper.className = "wgo-player-"+name+"-wrapper";
	e.element.appendChild(e.wrapper);
	parent.appendChild(e.element);
	if(!visible) e.element.style.display = "none";
	return e;
}

// generate all DOM of player
var BPgenerateDom = function() {
	// wrapper object for common DOM
	this.dom = {};
	
	// center element
	this.dom.center = document.createElement("div");
	this.dom.center.className = "wgo-player-center";
	
	// board wrapper element
	this.dom.board = document.createElement("div");
	this.dom.board.className = "wgo-player-board";
	
	// object wrapper for regions (left, right, top, bottom)
	this.regions = {};
	
	/*
	pseudo DOM structure:
	<main>
		<left></left>
		<center>
			<top></top>
			<board></board>
			<bottom></bottom>
		</center>
		<right></right>
	</main>
	*/
	
	this.regions.left = playerBlock("left", this.element);
	this.element.appendChild(this.dom.center);
	this.regions.right = playerBlock("right", this.element);
	
	this.regions.top = playerBlock("top", this.dom.center);
	this.dom.center.appendChild(this.dom.board);
	this.regions.bottom = playerBlock("bottom", this.dom.center);
}

var getCurrentLayout = function() {
	var cl = this.config.layout;
	if(cl.constructor != Array) return cl;
	
	var bh = this.height || this.maxHeight;
	for(var i = 0; i < cl.length; i++) {
		
		if(!cl[i].conditions || (
			(!cl[i].conditions.minWidth || cl[i].conditions.minWidth <= this.width) &&
			(!cl[i].conditions.minHeight || !bh || cl[i].conditions.minHeight <= bh) &&
			(!cl[i].conditions.maxWidth || cl[i].conditions.maxWidth >= this.width) &&
			(!cl[i].conditions.maxHeight || !bh || cl[i].conditions.maxHeight >= bh) &&
			(!cl[i].conditions.custom || cl[i].conditions.custom.call(this))
		  )) {
			return cl[i];
		}
	}
}

var appendComponents = function(area) {
	var components;
	
	if(this.currentLayout.layout) components = this.currentLayout.layout[area];
	else components = this.currentLayout[area];
	
	if(components) {
		this.regions[area].element.style.display = "block";
		
		for(var i in components) {
			if(!this.components[components[i]]) this.components[components[i]] = new BasicPlayer.component[components[i]](this);
			
			this.components[components[i]].appendTo(this.regions[area].wrapper);
			
			// remove detach flag
			this.components[components[i]]._detachFromPlayer = false;
		}
	}
	else {
		this.regions[area].element.style.display = "none";
	}

}

var manageComponents = function() {
	// add detach flags to every widget
	for(var key in this.components) {
		this.components[key]._detachFromPlayer = true;
	}
	
	appendComponents.call(this, "left");
	appendComponents.call(this, "right");
	appendComponents.call(this, "top");
	appendComponents.call(this, "bottom");
	
	// detach all invisible components
	for(var key in this.components) {
		if(this.components[key]._detachFromPlayer && this.components[key].element.parentNode) this.components[key].element.parentNode.removeChild(this.components[key].element);
	}
}

/**
 * Main object of player, it binds all magic together and produces visible player.
 * It inherits some functionality from WGo.PlayerView, but full html structure is done here.
 *
 * Layout of player can be set. It can be even dynamic according to screen resolution. 
 * There are 5 areas - left, right, top and bottom, and there is special region for board.
 * You can put BasicPlayer.Component objects to these regions. Basic components are: 
 *  - BasicPlayer.CommentBox - box with comments and game informations
 *  - BasicPlayer.InfoBox - box with information about players
 *  - BasicPlayer.Control - buttons and staff for control
 *
 * Possible configurations:
 *  - sgf: sgf string (default: undefined)
 *  - json: kifu stored in json/jgo (default: undefined)
 *  - sgfFile: sgf file path (default: undefined)
 *  - board: configuration object of board (default: {})
 *  - enableWheel: allow player to be controlled by mouse wheel (default: true)
 *  - lockScroll: disable window scrolling while hovering player (default: true)
 *  - enableKeys: allow player to be controlled by arrow keys (default: true)
 *  - kifuLoaded: extra Player's kifuLoaded event listener (default: undefined)
 *  - update: extra Player's update event listener (default: undefined)
 *  - frozen: extra Player's frozen event listener (default: undefined)
 *  - unfrozen: extra Player's unfrozen event listener (default: undefined)
 *  - layout: layout object. Look below how to define your own layout (default: BasicPlayer.dynamicLayout)
 *
 * You also must specify main DOMElement of player. 
 */

var BasicPlayer = WGo.extendClass(WGo.Player, function(elem, config) {
	this.config = config;
	
	// add default configuration of BasicPlayer
	for(var key in BasicPlayer.default) if(this.config[key] === undefined && BasicPlayer.default[key] !== undefined) this.config[key] = BasicPlayer.default[key];
	// add default configuration of Player class
	for(var key in WGo.Player.default) if(this.config[key] === undefined && WGo.Player.default[key] !== undefined) this.config[key] = WGo.Player.default[key];
	
	this.element = elem
	this.element.innerHTML = "";
	this.classes = (this.element.className ? this.element.className+" " : "")+"wgo-player-main" ;
	this.element.className = this.classes;
	if(!this.element.id) this.element.id = "wgo_"+(pl_count++);
	
	BPgenerateDom.call(this);
	
	this.board = new WGo.Board(this.dom.board, this.config.board);
	
	this.init();
	
	this.components = {};
	
	window.addEventListener("resize", function() {
		if(!this.noresize) {
			this.updateDimensions();
		}
		
	}.bind(this));
	
	this.updateDimensions();
	
	this.initGame();
});

/**
 * Append player to different element.
 */
 
BasicPlayer.prototype.appendTo = function(elem) {
	elem.appendChild(this.element);
	this.updateDimensions();
}

/**
 * Set right dimensions of all elements.
 */
	
BasicPlayer.prototype.updateDimensions = function() {
	var css = window.getComputedStyle(this.element);
	
	var els = [];
	while(this.element.firstChild) {
		els.push(this.element.firstChild);
		this.element.removeChild(this.element.firstChild);
	}
	
	var tmp_w = parseInt(css.width);
	var tmp_h = parseInt(css.height);
	var tmp_mh = parseInt(css.maxHeight) || 0;

	for(var i = 0; i < els.length; i++) {
		this.element.appendChild(els[i]);
	}

	if(tmp_w == this.width && tmp_h == this.height && tmp_mh == this.maxHeight) return;
	
	this.width = tmp_w;
	this.height = tmp_h;
	this.maxHeight = tmp_mh;

	this.currentLayout = getCurrentLayout.call(this);

	if(this.currentLayout && this.lastLayout != this.currentLayout) {
		if(this.currentLayout.className) this.element.className = this.classes+" "+this.currentLayout.className;
		else this.element.className = this.classes;
		manageComponents.call(this);
		this.lastLayout = this.currentLayout;
	}
	
	//var bw = this.width - this.regions.left.element.clientWidth - this.regions.right.element.clientWidth;
	var bw = this.dom.board.clientWidth;
	var bh = this.height || this.maxHeight;

	if(bh) {
		bh -= this.regions.top.element.offsetHeight + this.regions.bottom.element.offsetHeight;
	}
	
	if(bh && bh < bw) {
		if(bh != this.board.height) this.board.setHeight(bh);
	}
	else {
		if(bw != this.board.width) this.board.setWidth(bw);
	}
	
	var diff = bh - bw;
	
	if(diff > 0) {
		this.dom.board.style.height = bh+"px";
		this.dom.board.style.paddingTop = (diff/2)+"px";
	}
	else {
		this.dom.board.style.height = "auto";
		this.dom.board.style.paddingTop = "0";
	}
	
	this.regions.left.element.style.height = this.dom.center.offsetHeight+"px";
	this.regions.right.element.style.height = this.dom.center.offsetHeight+"px";

	for(var i in this.components) {
		if(this.components[i].updateDimensions) this.components[i].updateDimensions();
	}
}
	
/**
 * Layout contains built-in info box, for displaying of text(html) messages.
 * You can use this method to display a message.
 * 
 * @param text or html to display
 * @param closeCallback optional callback function which is called when message is closed
 */

BasicPlayer.prototype.showMessage = function(text, closeCallback, permanent) {
	this.info_overlay = document.createElement("div");
	this.info_overlay.style.width = this.element.offsetWidth+"px";
	this.info_overlay.style.height = this.element.offsetHeight+"px";
	this.info_overlay.className = "wgo-info-overlay";
	this.element.appendChild(this.info_overlay);
	
	var info_message = document.createElement("div");
	info_message.className = "wgo-info-message";
	info_message.innerHTML = text;
	
	var close_info = document.createElement("div");
	close_info.className = "wgo-info-close";
	if(!permanent) close_info.innerHTML = WGo.t("BP:closemsg");
	
	info_message.appendChild(close_info);
	
	this.info_overlay.appendChild(info_message);
	
	if(closeCallback) {
		this.info_overlay.addEventListener("click",function(e) {
			closeCallback(e);
		});
	}
	else if(!permanent) {
		this.info_overlay.addEventListener("click",function(e) {
			this.hideMessage();
		}.bind(this));
	}
	
	this.setFrozen(true);
}

/**
 * Hide a message box.
 */
 
BasicPlayer.prototype.hideMessage = function() {
	this.element.removeChild(this.info_overlay);
	this.setFrozen(false);
}

/**
 * Error handling
 */

BasicPlayer.prototype.error = function(err) {
	if(!WGo.ERROR_REPORT) throw err;
	
	var url = "#";
	
	switch(err.name) {
		case "InvalidMoveError": 
			this.showMessage("<h1>"+err.name+"</h1><p>"+err.message+"</p><p>If this message isn't correct, please report it by clicking <a href=\""+url+"\">here</a>, otherwise contact maintainer of this site.</p>");
		break;
		case "FileError":
			this.showMessage("<h1>"+err.name+"</h1><p>"+err.message+"</p><p>Please contact maintainer of this site. Note: it is possible to read files only from this host.</p>");
		break;
		default:
			this.showMessage("<h1>"+err.name+"</h1><p>"+err.message+"</p><pre>"+err.stacktrace+"</pre><p>Please contact maintainer of this site. You can also report it <a href=\""+url+"\">here</a>.</p>");
	}
} 

BasicPlayer.component = {};

/**
 * Preset layouts
 * They have defined regions as arrays, which can contain components. For each of these layouts each component specifies where it is placed.
 * You can create your own layout in same manners, but you must specify components manually.
 */
 
BasicPlayer.layouts = {
	"one_column": {
		top: [],
		bottom: [],
	},
	"no_comment": {
		top: [],
		bottom: [],
	},
	"right_top": {
		top: [],
		right: [],
	},
	"right": {
		right: [],
	},
	"minimal": {
		bottom: []
	},
};

/**
 * WGo player can have more layouts. It allows responsive design of the player.
 * Possible layouts are defined as array of object with this structure:
 * 
 * layout = {
 *   Object layout, // layout as specified above
 *   Object conditions, // conditions that has to be valid to apply this layout
 *   String className // custom classnames
 * }
 *
 * possible conditions:
 *  - minWidth - minimal width of player in px
 *  - maxWidth - maximal width of player in px
 *  - minHeight - minimal height of player in px
 *  - maxHeight - maximal height of player in px
 *  - custom - function which is called in template context, must return true or false
 *
 * Player's template evaluates layouts step by step and first layout that matches the conditions is applied.
 *
 * Look below at the default dynamic layout. Layouts are tested after every window resize.
 */

BasicPlayer.dynamicLayout = [
	{
		conditions: {
			minWidth: 650,
		},
		layout: BasicPlayer.layouts["right_top"], 
		className: "wgo-twocols wgo-large",
	},
	{
		conditions: {
			minWidth: 550,
			minHeight: 600,
		},
		layout: BasicPlayer.layouts["one_column"],
		className: "wgo-medium"
	},
	{
		conditions: {
			minWidth: 350,
		},
		layout: BasicPlayer.layouts["no_comment"],
		className: "wgo-small"
	},
	{	// if conditions object is omitted, layout is applied 
		layout: BasicPlayer.layouts["no_comment"],
		className: "wgo-xsmall",
	},
];

// default settings, they are merged with user settings in constructor.
BasicPlayer.default = {
	layout: BasicPlayer.dynamicLayout,
}

WGo.i18n.en["BP:closemsg"] = "click anywhere to close this window";

//--- Handling <div> with HTML5 data attributes -----------------------------------------------------------------

BasicPlayer.attributes = {
	"data-wgo": function(value) {
		if(value) {
			if(value[0] == "(") this.sgf = value;
			else this.sgfFile = value;
		}
	},
	
	"data-wgo-board": function(value) {
		// using eval to parse strings like "stoneStyle: 'painted'"
		this.board = eval("({"+value+"})");
	},
	
	"data-wgo-onkifuload": function(value) {
		this.kifuLoaded = new Function(value);
	},
	
	"data-wgo-onupdate": function(value) {
		this.update = new Function(value);
	},
	
	"data-wgo-onfrozen": function(value) {
		this.frozen = new Function(value);
	},
	
	"data-wgo-onunfrozen": function(value) {
		this.unfrozen = new Function(value);
	},
	
	"data-wgo-layout": function(value) {
		this.layout = eval("({"+value+"})");
	},
	
	"data-wgo-enablewheel": function(value) {
		if(value.toLowerCase() == "false") this.enableWheel = false;
	},
	
	"data-wgo-lockscroll": function(value) {
		if(value.toLowerCase() == "false") this.lockScroll = false;
	},
	
	"data-wgo-enablekeys": function(value) {
		if(value.toLowerCase() == "false") this.enableKeys = false;
	},
	
	"data-wgo-rememberpath": function(value) {
		if(value.toLowerCase() == "false") this.rememberPath = false;
	},
	
	"data-wgo-move": function(value) {
		var m = parseInt(value);
		if(m) this.move = m;
		else this.move = eval("({"+value+"})");
	},
}

var player_from_tag = function(elem) {
	var att, config, pl;
	
	config = {};

	for(var a = 0; a < elem.attributes.length; a++) {
		att = elem.attributes[a];
		if(BasicPlayer.attributes[att.name]) BasicPlayer.attributes[att.name].call(config, att.value, att.name);
	}

	pl = new BasicPlayer(elem, config);
	elem._wgo_player = pl;
}

WGo.BasicPlayer = BasicPlayer;

window.addEventListener("load", function() {
	var pl_elems = document.querySelectorAll("[data-wgo]");
	
	for(var i = 0; i < pl_elems.length; i++) {
		player_from_tag(pl_elems[i]);
	}
});

})(WGo);
