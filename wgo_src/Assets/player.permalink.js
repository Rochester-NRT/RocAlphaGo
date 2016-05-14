(function(WGo, undefined) {

"use strict";

var permalink = {
	active: true,
	query: {},
};

var handle_hash = function(player) {
	try {
		permalink.query = JSON.parse('{"'+window.location.hash.substr(1).replace('=', '":')+'}');
	}
	catch(e) {
		permalink.query = {};
	}
}

// add hashchange event
window.addEventListener("hashchange", function() {
	if(window.location.hash != "" && permalink.active) {
		handle_hash();

		for(var key in permalink.query) {
			var p_el = document.getElementById(key);
			if(p_el && p_el._wgo_player) p_el._wgo_player.goTo(move_from_hash);
		}
	}
});

// save hash query (after DOM is loaded - you can turn this off by setting WGo.Player.permalink.active = false;
window.addEventListener("DOMContentLoaded", function() {
	if(window.location.hash != "" && permalink.active) {
		handle_hash();
	}
});

// scroll into view of the board
window.addEventListener("load", function() {
	if(window.location.hash != "" && permalink.active) {
		for(var key in permalink.query) {
			var p_el = document.getElementById(key);
			if(p_el && p_el._wgo_player) {
				p_el.scrollIntoView();
				break;
			}
		}
	}
});

var move_from_hash = function() {
	if(permalink.query[this.element.id]) {
		return permalink.query[this.element.id].goto;
	}
}

WGo.Player.default.move = move_from_hash;

// add menu item
if(WGo.BasicPlayer && WGo.BasicPlayer.component.Control) {
	WGo.BasicPlayer.component.Control.menu.push({
		constructor: WGo.BasicPlayer.control.MenuItem,
		args: {
			name: "permalink",
			click: function(player) {
				var link = location.href.split("#")[0]+'#'+player.element.id+'={"goto":'+JSON.stringify(player.kifuReader.path)+'}';
				player.showMessage('<h1>'+WGo.t('permalink')+'</h1><p><input class="wgo-permalink" type="text" value=\''+link+'\' onclick="this.select(); event.stopPropagation()"/></p>');
			},
		}
	});
}

WGo.Player.permalink = permalink;
WGo.i18n.en["permalink"] = "Permanent link";

})(WGo);