(function() {

"use strict";

var prepare_dom = function() {
	prepare_dom_box.call(this,"white");
	prepare_dom_box.call(this,"black");
	this.element.appendChild(this.white.box);
	this.element.appendChild(this.black.box);
}

var prepare_dom_box = function(type) {
	this[type] = {};
	var t = this[type];
	t.box = document.createElement("div");
	t.box.className = "wgo-box-wrapper wgo-player-wrapper wgo-"+type;

	t.name = document.createElement("div");
	t.name.className = "wgo-box-title";
	t.name.innerHTML = type;
	t.box.appendChild(t.name);
	
	var info_wrapper;
	info_wrapper = document.createElement("div");
	info_wrapper.className = "wgo-player-info";
	t.box.appendChild(info_wrapper);
	
	t.info = {};
	t.info.rank = prepare_dom_info("rank");
	t.info.rank.val.innerHTML = "-";
	t.info.caps = prepare_dom_info("caps");
	t.info.caps.val.innerHTML = "0";
	t.info.time = prepare_dom_info("time");
	t.info.time.val.innerHTML = "--:--";
	info_wrapper.appendChild(t.info.rank.wrapper);
	info_wrapper.appendChild(t.info.caps.wrapper);
	info_wrapper.appendChild(t.info.time.wrapper);
}

var prepare_dom_info = function(type) {
	var res = {};
	res.wrapper = document.createElement("div");
	res.wrapper.className = "wgo-player-info-box-wrapper";
	
	res.box = document.createElement("div");
	res.box.className = "wgo-player-info-box";
	res.wrapper.appendChild(res.box);
	
	res.title = document.createElement("div");
	res.title.className = "wgo-player-info-title";
	res.title.innerHTML = WGo.t(type);
	res.box.appendChild(res.title);
	
	res.val = document.createElement("div");
	res.val.className = "wgo-player-info-value";
	res.box.appendChild(res.val);
	
	return res;
}

var kifu_loaded = function(e) {
	var info = e.kifu.info || {};
	
	if(info.black) {
		this.black.name.innerHTML = WGo.filterHTML(info.black.name) || WGo.t("Black");
		this.black.info.rank.val.innerHTML = WGo.filterHTML(info.black.rank) || "-";
	}
	else {
		this.black.name.innerHTML = WGo.t("Black");
		this.black.info.rank.val.innerHTML = "-";
	}
	if(info.white) {
		this.white.name.innerHTML = WGo.filterHTML(info.white.name) || WGo.t("White");
		this.white.info.rank.val.innerHTML = WGo.filterHTML(info.white.rank) || "-";
	}
	else {
		this.white.name.innerHTML = WGo.t("White");
		this.white.info.rank.val.innerHTML = "-";
	}
	
	this.black.info.caps.val.innerHTML = "0";
	this.white.info.caps.val.innerHTML = "0";
	
	if(info.TM) {
		this.setPlayerTime("black", info.TM);
		this.setPlayerTime("white", info.TM);
	}
	else {
		this.black.info.time.val.innerHTML = "--:--";
		this.white.info.time.val.innerHTML = "--:--";
	}
	
	this.updateDimensions();
}

var modify_font_size = function(elem) {
	var css, max, size;
	
	if(elem.style.fontSize) {
		var size = parseInt(elem.style.fontSize);
		elem.style.fontSize = "";
		css = window.getComputedStyle(elem);
		max = parseInt(css.fontSize);
		elem.style.fontSize = size+"px";
	}
	else {	
		css = window.getComputedStyle(elem);
		max = size = parseInt(css.fontSize);
	}
	
	if(size == max && elem.scrollHeight <= elem.offsetHeight) return;
	else if(elem.scrollHeight > elem.offsetHeight) {
		size -= 2;
		while(elem.scrollHeight > elem.offsetHeight && size > 1) {
			elem.style.fontSize = size+"px";
			size -= 2;
		}
	}
	else if(size < max) {
		size += 2;
		while(elem.scrollHeight <= elem.offsetHeight && size <= max) {
			elem.style.fontSize = size+"px";
			size += 2;
		}
		if(elem.scrollHeight > elem.offsetHeight) {
			elem.style.fontSize = (size-4)+"px";
		}
	}
}

var update = function(e) {
	if(e.node.BL) this.setPlayerTime("black", e.node.BL);
	if(e.node.WL) this.setPlayerTime("white", e.node.WL);
	if(e.position.capCount.black !== undefined) this.black.info.caps.val.innerHTML = e.position.capCount.black;
	if(e.position.capCount.white !== undefined) this.white.info.caps.val.innerHTML = e.position.capCount.white;
}

/**
 * Implements box with basic informations about go players.
 */

var InfoBox = WGo.extendClass(WGo.BasicPlayer.component.Component, function(player) {
	this.super(player);
	this.element.className = "wgo-infobox";
	
	prepare_dom.call(this);

	player.addEventListener("kifuLoaded", kifu_loaded.bind(this));
	player.addEventListener("update", update.bind(this));

});

InfoBox.prototype.setPlayerTime = function(color, time) {
	var min = Math.floor(time/60);
	var sec = Math.round(time)%60;
	this[color].info.time.val.innerHTML = min+":"+((sec < 10) ? "0"+sec : sec);
};

InfoBox.prototype.updateDimensions = function() {
	modify_font_size(this.black.name);
	modify_font_size(this.white.name);
};

var bp_layouts = WGo.BasicPlayer.layouts;
bp_layouts["right_top"].right.push("InfoBox");
bp_layouts["right"].right.push("InfoBox");
bp_layouts["one_column"].top.push("InfoBox");
bp_layouts["no_comment"].top.push("InfoBox");

WGo.i18n.en["rank"] = "Rank";
WGo.i18n.en["caps"] = "Caps";
WGo.i18n.en["time"] = "Time";

WGo.BasicPlayer.component.InfoBox = InfoBox;

})(WGo);
