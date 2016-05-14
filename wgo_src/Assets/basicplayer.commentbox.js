(function(WGo, undefined){

"use strict";

var prepare_dom = function() {
	this.box = document.createElement("div");
	this.box.className = "wgo-box-wrapper wgo-comments-wrapper";
	this.element.appendChild(this.box);
	
	this.comments_title = document.createElement("div");
	this.comments_title.className = "wgo-box-title";
	this.comments_title.innerHTML = WGo.t("comments");
	this.box.appendChild(this.comments_title);
	
	this.comments = document.createElement("div");
	this.comments.className = "wgo-comments-content";
	this.box.appendChild(this.comments);
	
	this.help = document.createElement("div");
	this.help.className = "wgo-help";
	this.help.style.display = "none";
	this.comments.appendChild(this.help);
	
	this.notification = document.createElement("div");
	this.notification.className = "wgo-notification";
	this.notification.style.display = "none";
	this.comments.appendChild(this.notification);
	
	this.comment_text = document.createElement("div");
	this.comment_text.className = "wgo-comment-text"; 
	this.comments.appendChild(this.comment_text);
}

var mark = function(move) {
	var x,y;
	
	x = move.charCodeAt(0)-'a'.charCodeAt(0);
	if(x < 0) x += 'a'.charCodeAt(0)-'A'.charCodeAt(0);
	if(x > 7) x--;
	y = (move.charCodeAt(1)-'0'.charCodeAt(0));
	if(move.length > 2) y = y*10+(move.charCodeAt(2)-'0'.charCodeAt(0));
	y = this.kifuReader.game.size-y;

	this._tmp_mark = {type:'MA', x:x, y:y};
	this.board.addObject(this._tmp_mark);
}

var unmark = function() {
	this.board.removeObject(this._tmp_mark);
	delete this._tmp_mark;
}

var search_nodes = function(nodes, player) {
	for(var i in nodes) {
		if(nodes[i].className && nodes[i].className == "wgo-move-link") {
			nodes[i].addEventListener("mouseover", mark.bind(player, nodes[i].innerHTML));
			nodes[i].addEventListener("mouseout", unmark.bind(player));
		}
		else if(nodes[i].childNodes && nodes[i].childNodes.length) search_nodes(nodes[i].childNodes, player);
	}
}	

var format_info = function(info, title) {
	var ret = '<div class="wgo-info-list">';
	if(title) ret += '<div class="wgo-info-title">'+WGo.t("gameinfo")+'</div>';
	for(var key in info) {
		ret += '<div class="wgo-info-item"><span class="wgo-info-label">'+key+'</span><span class="wgo-info-value">'+info[key]+'</span></div>';
	}
	ret += '</div>';
	return ret;
}

/**
 * Implements box for comments and game informations.
 */

var CommentBox = WGo.extendClass(WGo.BasicPlayer.component.Component, function(player) {
	this.super(player);
	this.player = player;
	
	this.element.className = "wgo-commentbox";
	
	prepare_dom.call(this);
	
	player.addEventListener("kifuLoaded", function(e) {
		if(e.kifu.hasComments()) {
			this.comments_title.innerHTML = WGo.t("comments");
			this.element.className = "wgo-commentbox";
			
			this._update = function(e) {
				this.setComments(e);
			}.bind(this);
			
			player.addEventListener("update", this._update);
		}
		else {
			this.comments_title.innerHTML = WGo.t("gameinfo");
			this.element.className = "wgo-commentbox wgo-gameinfo";
			
			if(this._update) {
				player.removeEventListener("update", this._update);
				delete this._update;
			}
			this.comment_text.innerHTML = format_info(e.target.getGameInfo());
		}
	}.bind(this));
	
	player.notification = function(text) {
		if(text) {
			this.notification.style.display = "block";
			this.notification.innerHTML = text;
			this.is_notification = true;
		}
		else {
			this.notification.style.display = "none";
			this.is_notification = false;
		}
		
		if(this.is_notification || this.is_help) this.comment_text.style.display = "none";
		else this.comment_text.style.display = "block";
		
	}.bind(this);
	
	player.help = function(text) {
		if(text) {
			this.help.style.display = "block";
			this.help.innerHTML = text;
			this.is_help = true;
		}
		else {
			this.help.style.display = "none";
			this.is_help = false;
		}
		
		if(this.is_notification || this.is_help) this.comment_text.style.display = "none";
		else this.comment_text.style.display = "block";
	}.bind(this);
});

CommentBox.prototype.setComments = function(e) {
	if(this.player._tmp_mark) unmark.call(this.player);

	var msg = "";
	if(!e.node.parent) {
		msg = format_info(e.target.getGameInfo(), true);
	}
	
	this.comment_text.innerHTML = msg+this.getCommentText(e.node.comment, this.player.config.formatNicks, this.player.config.formatMoves);

	if(this.player.config.formatMoves) {
		if(this.comment_text.childNodes && this.comment_text.childNodes.length) search_nodes(this.comment_text.childNodes, this.player);
	}
};

CommentBox.prototype.getCommentText = function(comment, formatNicks, formatMoves) {
	// to avoid XSS we must transform < and > to entities, it is highly recomanded not to change it
	//.replace(/</g,"&lt;").replace(/>/g,"&gt;") : "";
	if(comment) {
		var comm =  "<p>"+WGo.filterHTML(comment).replace(/\n/g, "</p><p>")+"</p>";
		if(formatNicks) comm = comm.replace(/(<p>)([^:]{3,}:)\s/g, '<p><span class="wgo-comments-nick">$2</span> ');
		if(formatMoves) comm = comm.replace(/\b[a-zA-Z]1?\d\b/g, '<a href="javascript:void(0)" class="wgo-move-link">$&</a>');
		return comm;
	}
	return "";
};

/**
 * Adding 2 configuration to BasicPlayer:
 *
 * - formatNicks: tries to highlight nicknames in comments (default: true)
 * - formatMoves: tries to highlight coordinates in comments (default: true)
 */
 
WGo.BasicPlayer.default.formatNicks = true;
WGo.BasicPlayer.default.formatMoves = true;

WGo.BasicPlayer.attributes["data-wgo-formatnicks"] = function(value) {
	if(value.toLowerCase() == "false") this.formatNicks = false;
}
	
WGo.BasicPlayer.attributes["data-wgo-formatmoves"] = function(value) {
	if(value.toLowerCase() == "false") this.formatMoves = false;
}

WGo.BasicPlayer.layouts["right_top"].right.push("CommentBox");
WGo.BasicPlayer.layouts["right"].right.push("CommentBox");
WGo.BasicPlayer.layouts["one_column"].bottom.push("CommentBox");

WGo.i18n.en["comments"] = "Comments";
WGo.i18n.en["gameinfo"] = "Game info";

WGo.BasicPlayer.component.CommentBox = CommentBox

})(WGo);
