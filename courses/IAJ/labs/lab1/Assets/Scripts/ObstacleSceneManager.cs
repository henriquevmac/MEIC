using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.IAJ.Unity.Movement;
using Assets.Scripts.IAJ.Unity.Movement.DynamicMovement;
using Assets.Scripts.IAJ.Unity.Util;
using UnityEngine;
using UnityEngine.UI;

public class ObstacleSceneManager : MonoBehaviour
{

    [Header("World Size Settings")]
    static public int X_WORLD_SIZE = 55;
    static public int Z_WORLD_SIZE = 32;

    [Header("Character to Clone")]
    public GameObject mainCharacter;

    [Header("Number of clones")]
    public int numberOfCharacters;

    [Header("Where to show information regarding movement")]
    public Text MovementText;
    private MainCharacterController mainCharacterController;
    private List<MainCharacterController> characterControllers;

    // Initalization of the scene
    void Start()
    {
        this.mainCharacterController = this.mainCharacter.GetComponent<MainCharacterController>();
        var textObj = GameObject.Find("InstructionsText");
        if (textObj != null)
        {
            textObj.GetComponent<Text>().text =
                "Instructions\n\n" +
                this.mainCharacterController.blendedKey + " - Blended\n" +
                this.mainCharacterController.priorityKey + " - Priority\n" +
                this.mainCharacterController.stopKey + " - Stop \n" +
                " Space - Change Target";
                
        }

        this.characterControllers = this.CloneCharacters(this.mainCharacter, numberOfCharacters);
        this.characterControllers.Add(this.mainCharacter.GetComponent<MainCharacterController>());

        //LINQ expression with a lambda function, returns an array with the DynamicCharacter for each secondary character controler
        var characters = this.characterControllers.Select(cc => cc.character).ToList();
        //add the character corresponding to the main character
        characters.Add(this.mainCharacterController.character);

        // Finding all of the obstacles in the World
        var obstacles = GameObject.FindGameObjectsWithTag("Obstacle");

        //initialize all the characters
        foreach (var characterController in this.characterControllers)
        {
            characterController.InitializeMovement(obstacles, characters);
        }
    }



    // Update is called once per frame
    void Update()
    {
        if (this.mainCharacterController.character.Movement != null)
        {
            this.MovementText.text = "Movement:" + this.mainCharacterController.character.Movement.Name;
        }
        else
        {
            MovementText.text = "Movement: ---";
        }
    }

    // Method to clone the 'objectToClone'
    private List<MainCharacterController> CloneCharacters(GameObject objectToClone, int numberOfCharacters)
    {
        var characters = new List<MainCharacterController>();
        var deltaAngle = MathConstants.MATH_2PI / numberOfCharacters;
        var angle = 0.0f + deltaAngle;

        for (int i = 1; i < numberOfCharacters; i++)
        {

            var clone = GameObject.Instantiate(objectToClone);
            var characterController = clone.GetComponent<MainCharacterController>();
                        
            // Some Quick Mafs to make sure clones spawn in a circle 
            characterController.character.KinematicData.Position = new Vector3(Mathf.Sin(angle) * 30, 0, -Mathf.Cos(angle) * 30);
            angle += deltaAngle;

            characters.Add(characterController);
        }

        return characters;
    }
}
