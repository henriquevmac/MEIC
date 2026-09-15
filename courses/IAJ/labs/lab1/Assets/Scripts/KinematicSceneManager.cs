using Assets.Scripts.IAJ.Unity.Movement.KinematicMovement;
using Assets.Scripts.IAJ.Unity.Util;
using UnityEngine;
using UnityEngine.UI;

public class KinematicSceneManager : MonoBehaviour
{
    [Header("World Size Settings")]
    public int X_WORLD_SIZE = 60; //55
    public int Z_WORLD_SIZE = 35; //32

    // You can but you don't need to mess with these
    private const float TIME_TO_TARGET = 2.0f;
    private const float RADIUS = 1.0f;

    // Characters
	private KinematicCharacter BlueCharacter { get; set; }
	private KinematicCharacter GreenCharacter { get; set; }


    // Debug text
    private Text BlueMovementText { get; set; }
    private Text GreenMovementText { get; set; }

	// Use this for initialization
	void Start () 
	{
        // Initializing the UI
		var textObj = GameObject.Find ("InstructionsText");
		if (textObj != null) 
		{
			textObj.GetComponent<Text>().text = 
				"Instructions\n\n" +
				"Blue Character\n" +
				"Q - Stationary\n" +
				"W - Seek\n" + 
				"E - Flee\n" + 
				"R - Arrive\n" + 
				"T - Wander\n\n" +  
				"Green Character\n" + 
				"A - Stationary\n" +
				"S - Seek\n" +
				"D - Flee\n" + 
				"F - Arrive\n" +
				"G - Wander\n"; 
		}

        //Finding the characters in the world
		var BlueObj = GameObject.Find ("Blue");
        if(BlueObj != null) this.BlueCharacter = this.GetOrCreateKinematicCharacter(BlueObj);
		var greenObj = GameObject.Find ("Green");
        if (greenObj != null) this.GreenCharacter = this.GetOrCreateKinematicCharacter(greenObj);

	    this.BlueMovementText = GameObject.Find("BlueMovement").GetComponent<Text>();
	    this.GreenMovementText = GameObject.Find("GreenMovement").GetComponent<Text>();
	}

	void Update()
	{

        //Dealing with input

		if (Input.GetKeyDown (KeyCode.Q)) 
		{
			this.BlueCharacter.Movement = null;
		} 
		else if (Input.GetKeyDown (KeyCode.W)) 
		{
            this.BlueCharacter.Movement = new KinematicSeek
			{
                Target = this.GreenCharacter.StaticData,
                MaxSpeed = this.BlueCharacter.MaxSpeed
			};
		}
		else if (Input.GetKeyDown (KeyCode.E)) 
		{
            this.BlueCharacter.Movement = new KinematicFlee
			{
                Target = this.GreenCharacter.StaticData,
                MaxSpeed = this.BlueCharacter.MaxSpeed
			};
		}
		else if (Input.GetKeyDown (KeyCode.R)) 
		{
            this.BlueCharacter.Movement = new KinematicArrive
			{
                Target = this.GreenCharacter.StaticData,
                MaxSpeed = this.BlueCharacter.MaxSpeed,
                TimeToTarget = TIME_TO_TARGET,
                Radius = RADIUS
			};
		}
		else if (Input.GetKeyDown (KeyCode.T)) 
		{
            this.BlueCharacter.Movement = new KinematicWander
			{
                MaxRotation = this.BlueCharacter.MaxRotation,
                MaxSpeed = this.BlueCharacter.MaxSpeed
			};
		}
        if (Input.GetKeyDown(KeyCode.A))
        {
            this.GreenCharacter.Movement = null;
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            this.GreenCharacter.Movement = new KinematicSeek
            {
                Target = this.BlueCharacter.StaticData,
                MaxSpeed = this.GreenCharacter.MaxSpeed
            };
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            this.GreenCharacter.Movement = new KinematicFlee
            {
                Target = this.BlueCharacter.StaticData,
                MaxSpeed = this.GreenCharacter.MaxSpeed
            };
        }
        else if (Input.GetKeyDown(KeyCode.F))
        {
            this.GreenCharacter.Movement = new KinematicArrive
            {
                Target = this.BlueCharacter.StaticData,
                MaxSpeed = this.GreenCharacter.MaxSpeed,
                TimeToTarget = TIME_TO_TARGET,
                Radius = RADIUS
            };
        }
        else if (Input.GetKeyDown(KeyCode.G))
        {
            this.GreenCharacter.Movement = new KinematicWander
            {
                MaxRotation = this.GreenCharacter.MaxRotation,
                MaxSpeed = this.GreenCharacter.MaxSpeed
            };
        }

        // Updating each of the characters
        this.UpdateMovingGameObject(this.BlueCharacter);
        this.UpdateMovingGameObject(this.GreenCharacter);

        this.UpdateMovementText();
	}

    private KinematicCharacter GetOrCreateKinematicCharacter(GameObject characterObj)
    {
        var character = characterObj.GetComponent<KinematicCharacter>();
        if (character == null)
        {
            character = characterObj.AddComponent<KinematicCharacter>();
            character.Initialize(characterObj);
        }

        return character;
    }

    private void UpdateMovingGameObject(KinematicCharacter movingCharacter)
    {
        if (movingCharacter.Movement != null)
        {
            movingCharacter.UpdateMovement();
            movingCharacter.StaticData.ApplyWorldLimit(X_WORLD_SIZE,Z_WORLD_SIZE);
            movingCharacter.GameObject.transform.position = movingCharacter.StaticData.Position;
        }
    }

    private void UpdateMovementText()
    {
        if (this.GreenCharacter.Movement == null)
        {
            this.GreenMovementText.text = "Green Movement: Stationary";
        }
        else
        {
            this.GreenMovementText.text = "Green Movement: " + this.GreenCharacter.Movement.Name;
        }

        if (this.BlueCharacter.Movement == null)
        {
            this.BlueMovementText.text = "Blue Movement: Stationary";
        }
        else
        {
            this.BlueMovementText.text = "Blue Movement: " + this.BlueCharacter.Movement.Name;
        }
    }
}
