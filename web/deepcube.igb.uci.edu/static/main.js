var state = [];
var rotateIdxs_old = null;
var rotateIdxs_new = null;
var stateToFE = null;
var FEToState = null;
var legalMoves = null;

var solveStartState = [];
var solveMoves = [];
var solveMoves_rev = [];
var solveIdx = null;
var solution_text = null;

// var faceNames = ["top", "bottom", "left", "right", "back", "front"];
var faceNames = ["top", "right", "front", "bottom", "left", "back"];
var colorMap = {
    0: "#ffffff",  // 白色
    1: "#ff0000",  // 红色
    2: "#00cc00",  // 绿色
    3: "#ffff00",  // 黄色
    4: "#ff9900",  // 橙色
    5: "#0000ff"   // 蓝色
};
var lastMouseX = 0,
  lastMouseY = 0;
var rotX = -30,
  rotY = -30;

var moves = []

var initState = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5
];

// 定义 idx 对换映射表
const idxSwapMap = {
    6: 0,
    0: 6,
    7: 1,
    1: 7,
    8: 2,
    2: 8,
    27: 33,
    33: 27,
    28: 34,
    34: 28,
    29: 35,
    35: 29
};

function mapIndex(idx) {
    return (idx in idxSwapMap) ? idxSwapMap[idx] : idx;
}

function reOrderArray(arr,indecies) {
	var temp = []
	for(var i = 0; i < indecies.length; i++) {
		var index = indecies[i]
		temp.push(arr[index])
	}

	return temp;
}

/*
	Rand int between min (inclusive) and max (exclusive)
*/
function randInt(min, max) {
	return Math.floor(Math.random() * (max - min)) + min;
}

function clearCube() {
  for (i = 0; i < faceNames.length; i++) {
    var myNode = document.getElementById(faceNames[i]);
    while (myNode.firstChild) {
      myNode.removeChild(myNode.firstChild);
    }
  }
}

function restoreCube() {
	setStickerColors(initState)
}

// function setStickerColors(newState) {
//     state = newState
//     clearCube()
//     idx = 0
//     for (i = 0; i < faceNames.length; i++) {
//         for (j = 0; j < 9; j++) {
//             var iDiv = document.createElement('div');
//             iDiv.className = 'sticker';
//             // 修正颜色索引获取方式
//             iDiv.style["background-color"] = colorMap[newState[idx]]
//             document.getElementById(faceNames[i]).appendChild(iDiv);
//             idx = idx + 1
//         }
//     }
// }

function setStickerColors(newState) {
    state = newState;
    clearCube();
    idx = 0;
    for (i = 0; i < faceNames.length; i++) {
        for (j = 0; j < 9; j++) {
            var iDiv = document.createElement('div');
            iDiv.className = 'sticker';

			swaped_idx = mapIndex(idx)

            // 设置颜色
            iDiv.style["background-color"] = colorMap[newState[swaped_idx]];

            // 在 sticker 上显示数字（idx）
            iDiv.textContent = swaped_idx;  // 显示在小方块里面
            iDiv.style.color = "black"; // 文字颜色
            iDiv.style.fontSize = "14px"; // 字体大小
            iDiv.style.textAlign = "center"; // 居中
            iDiv.style.lineHeight = "33.3%"; // 垂直居中

            document.getElementById(faceNames[i]).appendChild(iDiv);

            idx = idx + 1;
        }
    }
}

function buttonPressed(ev) {
	var face = ''
	var direction = ''

	if (ev.shiftKey) {
		direction = '_inv'
	}
	if (ev.which == 85 || ev.which == 117) {
		face='U'
	} else if (ev.which == 68 || ev.which == 100) {
		face = 'D'
	} else if (ev.which == 76 || ev.which == 108) {
		face = 'L'
	} else if (ev.which == 82 || ev.which == 114) {
		face = 'R'
	} else if (ev.which == 66 || ev.which == 98) {
		face = 'B'
	} else if (ev.which == 70 || ev.which == 102) {
		face = 'F'
	}
	if (face != '') {
		clearSoln();
		moves.push(face + direction);
		nextState();
	}
}


function enableScroll() {
	document.getElementById("first_state").disabled=false;
	document.getElementById("prev_state").disabled=false;
	document.getElementById("next_state").disabled=false;
	document.getElementById("last_state").disabled=false;
}

function disableScroll() {
	document.getElementById("first_state").blur(); //so keyboard input can work without having to click away from disabled button
	document.getElementById("prev_state").blur();
	document.getElementById("next_state").blur();
	document.getElementById("last_state").blur();

	document.getElementById("first_state").disabled=true;
	document.getElementById("prev_state").disabled=true;
	document.getElementById("next_state").disabled=true;
	document.getElementById("last_state").disabled=true;
}

/*
	Clears solution as well as disables scroll
*/
function clearSoln() {
	solveIdx = 0;
	solveStartState = [];
	solveMoves = [];
	solveMoves_rev = [];
	solution_text = null;
	document.getElementById("solution_text").innerHTML = "Solution:";
	disableScroll();
}

function setSolnText(setColor=true) {
	solution_text_mod = JSON.parse(JSON.stringify(solution_text))
	if (solveIdx >= 0) {
		if (setColor == true) {
			solution_text_mod[solveIdx] = solution_text_mod[solveIdx].bold().fontcolor("blue")
		} else {
			solution_text_mod[solveIdx] = solution_text_mod[solveIdx]
		}
	}
	document.getElementById("solution_text").innerHTML = "Solution: "+ solution_text_mod.join(" ");
}

function enableInput() {
	document.getElementById("scramble").disabled=false;
	document.getElementById("solve").disabled=false;
	document.getElementById("clear").disabled=false;
	$(document).on("keypress", buttonPressed);
}

function disableInput() {
	document.getElementById("scramble").disabled=true;
	document.getElementById("solve").disabled=true;
	$(document).off("keypress", buttonPressed);
}

function nextState(moveTimeout=0) {
	if (moves.length > 0) {
		disableInput();
		disableScroll();
		move = moves.shift() // get Move

		// 添加安全检查
        if (!rotateIdxs_new || !rotateIdxs_new[move]) {
            console.error('Invalid move or rotateIdxs_new not initialized:', move);
            enableInput();
            return;
        }
		
		//convert to python representation
		state_rep = reOrderArray(state,FEToState)
		newState_rep = JSON.parse(JSON.stringify(state_rep))

		//swap stickers
		for (var i = 0; i < rotateIdxs_new[move].length; i++) {
			newState_rep[rotateIdxs_new[move][i]] = state_rep[rotateIdxs_old[move][i]]
		}

		// Change move highlight
		if (moveTimeout != 0){ //check if nextState is used for first_state click, prev_state,etc.
				solveIdx++
				setSolnText(setColor=true)
		}

		//convert back to HTML representation
		newState = reOrderArray(newState_rep,stateToFE)

		//set new state
		setStickerColors(newState)

		//Call again if there are more moves
		if (moves.length > 0) {
			setTimeout(function(){nextState(moveTimeout)}, moveTimeout);
		} else {
			enableInput();
			if (solveMoves.length > 0) {
				enableScroll();
				setSolnText();
			}
		}
	} else {
		enableInput();
		if (solveMoves.length > 0) {
			enableScroll();
			setSolnText();
		}
	}
}

function scrambleCube() {
	disableInput();
	clearSoln();

	numMoves = randInt(100,200);
	for (var i = 0; i < numMoves; i++) {
		moves.push(legalMoves[randInt(0,legalMoves.length)]);
	}

	nextState(0);
}

function solveCube() {
    disableInput();
    clearSoln();
    document.getElementById("solution_text").innerHTML = "SOLVING..."
    $.ajax({
        url: '/solve',
        data: JSON.stringify({"state": state}),
        type: 'POST',
        contentType: 'application/json',
        dataType: 'json',
        // timeout: 5000,
        success: function(response) {
            if (response.error) {
                // 处理业务逻辑错误
                document.getElementById("solution_text").innerHTML = "Error: " + response.error;
                enableInput();
            } else {
                // 正常处理成功响应
                solveStartState = JSON.parse(JSON.stringify(state))
                solveMoves = response["moves"];
                solveMoves_rev = response["moves_rev"];
                solution_text = response["solve_text"];
                solution_text.push("SOLVED!")
                setSolnText(true);

                moves = JSON.parse(JSON.stringify(solveMoves))

                setTimeout(function(){nextState(500)}, 500);
            }
        },
        error: function(xhr, status, error) {
            // 处理HTTP请求错误
            console.log("AJAX Error:", status, error);
            var errorMessage = "请求失败，请重试";
            if (status === "timeout") {
                errorMessage = "请求超时，请重试";
            } else if (xhr.status === 404) {
                errorMessage = "未找到解决方案";
            } else if (xhr.status === 500) {
                errorMessage = "服务器内部错误，请稍后再试";
            } else if (xhr.status === 400) {
                errorMessage = "请求参数错误";
            }
            document.getElementById("solution_text").innerHTML = errorMessage;
            enableInput();
        }
    });
}

$( document ).ready($(function() {
	disableInput();
	clearSoln();
	$.ajax({
		url: '/initState',
		data: {},
		type: 'POST',
		dataType: 'json',
		success: function(response) {
			setStickerColors(response["state"]);
			rotateIdxs_old = response["rotateIdxs_old"];
			rotateIdxs_new = response["rotateIdxs_new"];
			stateToFE = response["stateToFE"];
			FEToState = response["FEToState"];
			legalMoves = response["legalMoves"]
			enableInput();
		},
		error: function(error) {
			console.log(error);
		},
	});

	$("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)"); //Initial orientation	

	$('#scramble').click(function() {
		scrambleCube()
	});

	$('#solve').click(function() {
		solveCube()
	});

	$('#clear').click(function() {
		restoreCube()
	})

	$('#first_state').click(function() {
		if (solveIdx > 0) {
			moves = solveMoves_rev.slice(0, solveIdx).reverse();
			solveIdx = 0;
			nextState();
		}
	});

	$('#prev_state').click(function() {
		if (solveIdx > 0) {
			solveIdx = solveIdx - 1
			moves.push(solveMoves_rev[solveIdx])
			nextState()
		}
	});

	$('#next_state').click(function() {
		if (solveIdx < solveMoves.length) {
			moves.push(solveMoves[solveIdx])
			solveIdx = solveIdx + 1
			nextState()
		}
	});

	$('#last_state').click(function() {
		if (solveIdx < solveMoves.length) {
			moves = solveMoves.slice(solveIdx, solveMoves.length);
			solveIdx = solveMoves.length
			nextState();
		}
	});

	$('#cube_div').on("mousedown", function(ev) {
		lastMouseX = ev.clientX;
		lastMouseY = ev.clientY;
		$('#cube_div').on("mousemove", mouseMoved);
	});
	$('#cube_div').on("mouseup", function() {
		$('#cube_div').off("mousemove", mouseMoved);
	});
	$('#cube_div').on("mouseleave", function() {
		$('#cube_div').off("mousemove", mouseMoved);
	});

	console.log( "ready!" );
}));


function mouseMoved(ev) {
  var deltaX = ev.pageX - lastMouseX;
  var deltaY = ev.pageY - lastMouseY;

  lastMouseX = ev.pageX;
  lastMouseY = ev.pageY;

  rotY += deltaX * 0.2;
  rotX -= deltaY * 0.5;

  $("#cube").css("transform", "translateZ( -100px) rotateX( " + rotX + "deg) rotateY(" + rotY + "deg)");
}

