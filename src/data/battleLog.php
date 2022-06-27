<?php

if(! isset($_POST)){
	$response = [
		'response' => 'error'
		];

	echo json_encode($response);
	
	exit();
}

$results = $_POST['results'];

if (!($file = fopen('battleLog.csv', "a+"))){
	$response = [
		'response' => 400,
		'message' => 'error creating filestream'
	];
}
if (fwrite($file, $results) === FALSE){
	$response = [
		'response' => 400,
		'message' => 'error writing to file'
	];
}
fclose($file);

$response = ['response' => 'success', 'data' => $results];
echo json_encode($response);
?>