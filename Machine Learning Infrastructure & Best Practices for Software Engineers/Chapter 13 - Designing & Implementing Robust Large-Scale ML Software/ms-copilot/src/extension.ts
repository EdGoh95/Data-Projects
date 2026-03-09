// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext)
{
	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "ms-copilot" is now active!');

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand('ms-copilot.get_suggestion', async () => 
    {
		// The code you place here will be executed every time your command is executed
		// Display a message box to the user
        let x:string = 'Result: ';
		vscode.window.showInformationMessage('Hello World from ms-copilot!');

        const fetch = require('node-fetch');
        const result = await fetch('https://www.google.com');
        const txt = await result.text();
        x+= txt;

        vscode.window.showInformationMessage(x);
	});

    // Define a command to check which code is selected.
    vscode.commands.registerCommand('ms-copilot.logSelectedText', () => 
    {
        // Libraries needed to execute the python scripts
        const python = require('python-shell');
        const path = require('path');

        // Set up the path to the right python interpreter in case we have a virtual environment
        python.PythonShell.defaultOptions = {pythonPath: '/opt/anaconda3/envs/PyTorch/bin/python'};
        // Get the active text editor
        const editor = vscode.window.activeTextEditor;
        // Get the selected text
        const selectedText = editor!.document.getText(editor!.selection);
        
        // Prompt is the same as the selected text
        let prompt:string = selectedText;
        // This is the Python script that we execute to get the generated code from the Parrot model
        // Note the strange formatting which is necessary since Python is sensitive to indentation
        let scriptText = `
from transformers import pipeline
pipe = pipeline('text-generation', model = 'codeparrot/codeparrot-small')
outputs = pipe("${prompt}", max_new_token = 30, do_sample = False)
print(outputs[0]['generated_text'])`;

        // Let the user know when the code generation has started
        vscode.window.showInformationMessage(`Starting code generation for prompt: ${prompt}`);
        // Run the script and get back the message
        python.PythonShell.runString(scriptText, null).then(messages  => 
        {
            console.log(messages);
            
            // Paste the code in the active editor
            let activeEditor = vscode.window.activeTextEditor;
            activeEditor!.edit((selectedText) => 
            {
                // Format the response as a single string, not an array of strings
                let snippet = messages.join('\n');
                // Replace the selected text with the output
                selectedText.replace(activeEditor!.selection, snippet);

                // Logging the content of the snippet to the console
                console.log(snippet);
            });
        }).then(() => 
        {
            vscode.window.showInformationMessage(`Code generation completed!`);});
        });

	context.subscriptions.push(disposable);
}

// This method is called when your extension is deactivated
export function deactivate() {}
