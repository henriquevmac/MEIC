using Assets.Scripts.IAJ.Unity.Movement;
using Assets.Scripts.IAJ.Unity.Movement.DynamicMovement;
using UnityEngine;
using UnityEngine.UI;

public class DynamicSceneManager : MonoBehaviour {

    [Header("World Size Settings")]
    public int X_WORLD_SIZE = 55;
    public int Z_WORLD_SIZE = 32;

	public DynamicCharacter BlueCharacter { get; set; }
	public DynamicCharacter GreenCharacter { get; set; }

    private Text BlueMovementText { get; set; }
    private Text GreenMovementText { get; set; }

    private DynamicWander BlueDynamicWander { get; set; }
    private DynamicWander GreenDynamicWander { get; set; }
    private DynamicSeek BlueDynamicSeek { get; set; }
    private DynamicSeek GreenDynamicSeek { get; set; }
    private DynamicFlee BlueDynamicFlee { get; set; }
    private DynamicFlee GreenDynamicFlee { get; set; }

    [Header("Debug objects")]
    public GameObject blueDebugTarget;
    public GameObject greenDebugTarget;


	// Use this for initialization
	void Start () 
	{
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

		var blueObj = GameObject.Find ("Blue");
        if(blueObj != null) this.BlueCharacter = this.GetOrCreateDynamicCharacter(blueObj);
		var greenObj = GameObject.Find ("Green");
        if (greenObj != null) this.GreenCharacter = this.GetOrCreateDynamicCharacter(greenObj);

	    this.BlueMovementText = GameObject.Find("BlueMovement").GetComponent<Text>();
	    this.GreenMovementText = GameObject.Find("GreenMovement").GetComponent<Text>();

        #region movement initialization

        this.BlueDynamicSeek = new DynamicSeek
        {
            Character = this.BlueCharacter,
            Target = this.GreenCharacter.KinematicData
        };

        this.BlueDynamicFlee = new DynamicFlee
		{
            Character = this.BlueCharacter,
			Target = this.GreenCharacter.KinematicData
		};

	    this.BlueDynamicWander = new DynamicWander
	    {
            Character = this.BlueCharacter,
            DebugTarget = this.blueDebugTarget
            // Are there any variables the need to be initialized?
        };

        this.GreenDynamicSeek = new DynamicSeek
        {
            Character = this.GreenCharacter,
            Target = this.BlueCharacter.KinematicData
        };

        this.GreenDynamicFlee = new DynamicFlee
        {
            Character = this.GreenCharacter,
            Target = this.BlueCharacter.KinematicData
        };

        // Are there any variables the need to be initialized?
        this.GreenDynamicWander = new DynamicWander()
        {
            Character = this.GreenCharacter,
            DebugTarget = this.greenDebugTarget
        };

        #endregion
    }

	void Update()
	{
		if (Input.GetKeyDown (KeyCode.Q)) 
		{
			this.BlueCharacter.Movement = null;
		} 
		else if (Input.GetKeyDown (KeyCode.W))
		{
		    this.BlueCharacter.Movement = this.BlueDynamicSeek;
		}
		else if (Input.GetKeyDown (KeyCode.E))
		{
		    this.BlueCharacter.Movement = this.BlueDynamicFlee;
		}
		
		else if (Input.GetKeyDown (KeyCode.T))
		{
            // TODO 

        }


        if (Input.GetKeyDown(KeyCode.A))
        {
            this.GreenCharacter.Movement = null;
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            this.GreenCharacter.Movement = this.GreenDynamicSeek;
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            this.GreenCharacter.Movement = this.GreenDynamicFlee;
        }
        
        else if (Input.GetKeyDown(KeyCode.G))
        {
            // TODO 
        }


        this.UpdateMovingGameObject(this.BlueCharacter);
        this.UpdateMovingGameObject(this.GreenCharacter);

	    if (this.blueDebugTarget != null && this.BlueCharacter.Movement != null)
	    {
	        this.blueDebugTarget.transform.position = this.BlueCharacter.Movement.Target.Position;
	    }

	    if (this.greenDebugTarget != null && this.GreenCharacter.Movement != null)
	    {
	        this.greenDebugTarget.transform.position = this.GreenCharacter.Movement.Target.Position;
	    }

	    this.UpdateMovementText();
	}

    private DynamicCharacter GetOrCreateDynamicCharacter(GameObject characterObj)
    {
        var character = characterObj.GetComponent<DynamicCharacter>();
        if (character == null)
        {
            character = characterObj.AddComponent<DynamicCharacter>();
            character.Initialize(characterObj);
        }

        return character;
    }

    private void UpdateMovingGameObject(DynamicCharacter movingCharacter)
    {
        if (movingCharacter.Movement != null)
        {
            movingCharacter.UpdateMovement();
            movingCharacter.KinematicData.ApplyWorldLimit(X_WORLD_SIZE,Z_WORLD_SIZE);
            movingCharacter.GameObject.transform.position = movingCharacter.KinematicData.Position;
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
