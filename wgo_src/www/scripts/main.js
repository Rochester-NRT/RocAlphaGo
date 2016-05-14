(function(_WIN) {
    function wGame(containerElement, boardConfig) {
        var _step = 0;
        var _myColor = WGo.B;
        var _isMyTurn = false;
        var _board;
        var maxBorder = 19;
        var sessionId = '';

        function numberToAlphabet(numX, numY) {
            if (numX >= maxBorder || numY >= maxBorder || numX < 0 || numY < 0) {
                return;
            }
            var base = 'a'.charCodeAt();
            var result = String.fromCharCode(numX + base) + String.fromCharCode(numY + base);
            return result;
        }

        function alphabetToNumber(strRes) {
            if (strRes.length != 2) {
                return;
            }
            strRes = strRes.toLowerCase();
            var base = 'a'.charCodeAt();
            var numX = strRes[0].charCodeAt() - base;
            var numY = strRes[1].charCodeAt() - base;
            if (numX < 0 || numY < 0 || numX >= maxBorder || numY >= maxBorder) {
                return;
            }
            return [numX, numY];
        }

        function getWgoColorString(colorValue) {
            return colorValue > 0 ? "Black" : "White";
        }

        function getWgoColor(argument) {
            return _myColor === WGo.B ? WGo.W : WGo.B;
        }

        function logWgo(step, color, x, y) {
            console.log('Step: %s, Color: %s, Point: %s, %s', step, getWgoColorString(color), x, y);
        }

        var Point = function(x, y) {
            this.x = x;
            this.y = y;
        };
        var userInfo = {
            name: '',
            rank: ''
        };
        // Robot round
        // Return new board
        var userStep = new Point(-1, -1);
        userInfo.name = 'test';
        userInfo.rank = '0';

        function opponentRound() {
            _isMyTurn = false;
            var url = 'http://geekgo.corp.microsoft.com:8080/move';
            var wgoColor = getWgoColor(_myColor);

            function successed(wgoStep) {
                _step++;

                logWgo(_step, wgoColor, wgoStep.x, wgoStep.y);

                sessionId = wgoStep.sessionId;

                // remove dead stones
                if (wgoStep.dead_stones) {
                    wgoStep.dead_stones.map(function(removeAction) {
                        _board.removeObjectsAt(removeAction[0], removeAction[1]);
                    });
                }

                _board.addObject({
                    x: wgoStep.x,
                    y: wgoStep.y,
                    c: wgoColor
                });

                _isMyTurn = true;
            }

            function failed() {
                console.log('There is something wrong with server');
            }

            var postData = {
                sessionId: sessionId,
                name: userInfo.name,
                rank: userInfo.rank,
                x: userStep.x,
                y: userStep.y
            };
            $.ajax({
                    url: url,
                    data: JSON.stringify(postData),
                    type: 'POST',
                    contentType: "application/json"
                })
                .done(successed)
                .fail(failed);
        }

        function myRound(x, y) {
            _step++;
            logWgo(_step, _myColor, x, y);

            userStep.x = x;
            userStep.y = y;

            // Add my stone to board
            _board.addObject({
                x: x,
                y: y,
                c: _myColor
            });
            opponentRound();
        }

        var self = {
            init: function() {
                var _this = this;

                // Create board
                _board = new WGo.Board(containerElement, boardConfig);
                _board.addEventListener("click", function(x, y) {
                    if (!_isMyTurn) {
                        return;
                    }
                    myRound(x, y);
                });

                // 0: black
                // 1: white
                _myColor = Math.floor(Math.random() * 2) ? WGo.W : WGo.B;
                console.log("My color: " + getWgoColorString(_myColor));
                if (_myColor === WGo.B) {
                    _isMyTurn = true;
                } else {
                    // Robot round
                    opponentRound();
                }
            },
            getMyColor: function() {
                return _myColor;
            },
            getStep: function() {
                return _step;
            },
            getIsMyTurn: function() {
                return _isMyTurn;
            }
        };

        return self;
    }

    _WIN.wGame = wGame;
})(window);

var game = wGame(document.getElementById("board"), {
width: 400,
});
game.init();


