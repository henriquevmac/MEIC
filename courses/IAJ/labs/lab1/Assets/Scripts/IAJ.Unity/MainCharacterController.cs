using Assets.Scripts.IAJ.Unity.Util;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using System.Collections.Generic;
using Assets.Scripts.IAJ.Unity.Movement.DynamicMovement;
using Assets.Scripts.IAJ.Unity.Movement;
using Assets.Scripts.IAJ.Unity.Movement.Arbitration;

public class MainCharacterController : MonoBehaviour
{
    // Default values
    private float worldSizeX = 55;
    private float worldSizeZ = 32.5f;

    public KeyCode stopKey = KeyCode.S;
    public KeyCode priorityKey = KeyCode.P;
    public KeyCode blendedKey = KeyCode.B;

    public DynamicCharacter character;

    public PriorityMovement priorityMovement;
    public BlendedMovement blendedMovement;
    private DynamicPatrol patrolMovement;


    //Initialization
    void Awake()
    {
        this.character = this.GetComponent<DynamicCharacter>();
        if (this.character == null)
        {
            this.character = this.gameObject.AddComponent<DynamicCharacter>();
        }

        worldSizeX = ObstacleSceneManager.X_WORLD_SIZE;
        worldSizeZ = ObstacleSceneManager.Z_WORLD_SIZE;


        this.priorityMovement = new PriorityMovement
        {
            Character = this.character
        };

        this.blendedMovement = new BlendedMovement
        {
            Character = this.character
        };
    }


    public void InitializeMovement(GameObject[] obstacles, List<DynamicCharacter> characters)
    {
        foreach (var obstacle in obstacles)
        {
            // Adjust the parameters below to fine-tune the behaviour...
            var avoidObstacleMovement = new DynamicAvoidObstacle(obstacle)
            {
                Character = this.character   
            };

            // Adjust the blended weight...
            this.blendedMovement.Movements.Add(new MovementWithWeight(avoidObstacleMovement, 5.0f));
            this.priorityMovement.Movements.Add(avoidObstacleMovement);
            //Note: This implementation (one avoidObstacleMovement per obstacle for each character is very inefficient). It is possible to implement this as
            // a single avoidObstacleMovement using a general Raycast, instead of Collider.Raycast... 
        }

        foreach (var otherCharacter in characters)
        {
            if (otherCharacter != this.character)
            {
                //TODO: add your AvoidCharacter movement here. Do you need to define any parameters?
                
                //TODO Add it to the Blend and Priority Movement here
                

            }
        }

        var targetPosition = this.character.KinematicData.Position + (Vector3.zero - this.character.KinematicData.Position) * 2;

        this.patrolMovement = new DynamicPatrol(this.character.KinematicData.Position, targetPosition)
        {
            Character = this.character,
        };

        this.priorityMovement.Movements.Add(patrolMovement);
        this.blendedMovement.Movements.Add(new MovementWithWeight(patrolMovement, 1));
        this.character.Movement = this.priorityMovement;
    }


    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            this.patrolMovement.ChangeTarget();
        }
        if (Input.GetKeyDown(this.stopKey))
        {
            this.character.Movement = null;
        }
        else if (Input.GetKeyDown(this.blendedKey))
        {
            this.character.Movement = this.blendedMovement;
        }
        else if (Input.GetKeyDown(this.priorityKey))
        {
            this.character.Movement = this.priorityMovement;
        }

        this.UpdateMovingGameObject();
    }

    private void UpdateMovingGameObject()
    {
        if (this.character.Movement != null)
        {
            this.character.UpdateMovement();
            this.character.KinematicData.ApplyWorldLimit(this.worldSizeX, this.worldSizeZ);
        }
    }
}
