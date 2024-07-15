import {app} from "../../../scripts/app.js"

function innerText(node,text){
    const textWidget = node.widgets.find((w)=>w.name === "text")
    textWidget.value = text
}

app.registerExtension({
    name: "SenseVoice.innerText",
    async beforeRegisterNodeDef(nodeType,nodeData,app){
        if (nodeData?.name == "TextNode"){
            nodeType.prototype.onExecuted = function(data){
                console.log(data.text[0]);
                innerText(this,data.text[0])
            }
        }
    }
})