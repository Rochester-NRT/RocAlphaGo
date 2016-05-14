(function(WGo, undefined) {

"use strict";

/**
 * Base class for BasicPlayer's component. Each component should implement this interface.
 */

var Component = function() {
	this.element = document.createElement("div");
}

Component.prototype = {
	constructor: Component,
	
	/**
	 * Append component to element.
	 */
	
	appendTo: function(target) {
		target.appendChild(this.element);
	},
	
	/**
	 * Compute and return width of component.
	 */
	 
	getWidth: function() {
		var css = window.getComputedStyle(this.element);
		return parseInt(css.width);
	},
	
	/**
	 * Compute and return height of component.
	 */
	
	getHeight: function() {
		var css = window.getComputedStyle(this.element);
		return parseInt(css.height);
	},
	
	/**
	 * Update component. Actually dimensions are defined and cannot be changed in this method, 
	 * but you can change for example font size according to new dimensions.
	 */
	
	updateDimensions: function() {
	
	}
}

WGo.BasicPlayer.component.Component = Component;

})(WGo);