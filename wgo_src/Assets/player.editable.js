
(function(WGo) {

// board mousemove callback for edit move - adds highlighting
var edit_board_mouse_move = function(x,y) {
	if(this.player.frozen || (this._lastX == x && this._lastY == y)) return;
	
	this._lastX = x;
	this._lastY = y;
	
	if(this._last_mark) {
		this.board.removeObject(this._last_mark);
	}
	
	if(x != -1 && y != -1 && this.player.kifuReader.game.isValid(x,y)) {
		this._last_mark = {
			type: "outline",
			x: x,
			y: y, 
			c: this.player.kifuReader.game.turn
		};
		this.board.addObject(this._last_mark);
	}
	else {
		delete this._last_mark;
	}
}

// board mouseout callback for edit move	
var edit_board_mouse_out = function() {
	if(this._last_mark) {
		this.board.removeObject(this._last_mark);
		delete this._last_mark;
		delete this._lastX;
		delete this._lastY;
	}
}

// get differences of two positions as a change object (TODO create a better solution, without need of this function)
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

WGo.Player.Editable = {};

/**
 * Toggle edit mode.
 */
	
WGo.Player.Editable = function(player, board) {
	this.player = player;
	this.board = board;
	this.editMode = false;
}

WGo.Player.Editable.prototype.set = function(set) {
	if(!this.editMode && set) {
		// save original kifu reader
		this.originalReader = this.player.kifuReader;
		
		// create new reader with cloned kifu
		this.player.kifuReader = new WGo.KifuReader(this.player.kifu.clone(), this.originalReader.rememberPath);
		
		// go to current position
		this.player.kifuReader.goTo(this.originalReader.path);
		
		// register edit listeners
		this._ev_click = this._ev_click || this.play.bind(this);
		this._ev_move = this._ev_move || edit_board_mouse_move.bind(this);
		this._ev_out = this._ev_out || edit_board_mouse_out.bind(this);
		
		this.board.addEventListener("click", this._ev_click);
		this.board.addEventListener("mousemove", this._ev_move);
		this.board.addEventListener("mouseout", this._ev_out);
		
		this.editMode = true;
	}
	else if(this.editMode && !set) {
		// go to the last original position
		this.originalReader.goTo(this.player.kifuReader.path);
		
		// change object isn't actual - update it, not elegant solution, but simple
		this.originalReader.change = pos_diff(this.player.kifuReader.getPosition(), this.originalReader.getPosition());
		
		// update kifu reader
		this.player.kifuReader = this.originalReader;
		this.player.update(true);
		
		// remove edit listeners
		this.board.removeEventListener("click", this._ev_click);
		this.board.removeEventListener("mousemove", this._ev_move);
		this.board.removeEventListener("mouseout", this._ev_out);
		
		this.editMode = false;
	}
}

WGo.Player.Editable.prototype.play = function(x,y) {
	if(this.player.frozen || !this.player.kifuReader.game.isValid(x, y)) return;
	
	this.player.kifuReader.node.appendChild(new WGo.KNode({
		move: {
			x: x, 
			y: y, 
			c: this.player.kifuReader.game.turn
		}, 
		edited: true
	}));
	this.player.next(this.player.kifuReader.node.children.length-1);
}

if(WGo.BasicPlayer && WGo.BasicPlayer.component.Control) {
	WGo.BasicPlayer.component.Control.menu.push({
		constructor: WGo.BasicPlayer.control.MenuItem,
		args: {
			name: "editmode",
			togglable: true,
			click: function(player) { 
				this._editable = this._editable || new WGo.Player.Editable(player, player.board);
				this._editable.set(!this._editable.editMode);
				return this._editable.editMode;
			},
			init: function(player) {
				var _this = this;
				player.addEventListener("frozen", function(e) {
					_this._disabled = _this.disabled;
					if(!_this.disabled) _this.disable();
				});
				player.addEventListener("unfrozen", function(e) {
					if(!_this._disabled) _this.enable();
					delete _this._disabled;
				});
			},
		}
	}); 
}

WGo.i18n.en["editmode"] = "Edit mode";

})(WGo);
