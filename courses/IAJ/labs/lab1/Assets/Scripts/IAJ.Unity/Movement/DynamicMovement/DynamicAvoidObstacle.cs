using Assets.Scripts.IAJ.Unity.Util;
using System;
using UnityEngine;

namespace Assets.Scripts.IAJ.Unity.Movement.DynamicMovement
{
    public class DynamicAvoidObstacle : DynamicSeek
    {
        //This implementation detects and avoids collisions with a specified obstacle, whose collider is stored in ObstacleCollider;
        //The advantage as it allows us to test collisions with only the objects that interest us, but is very inneficient when
        //there are many objects and characters on the scene...
        public override string Name
        {
            get { return "Avoid Obstacle"; }
        }

        private GameObject obstacle;

        public GameObject Obstacle
        {
            get { return this.obstacle; }
            set
            {
                this.obstacle = value;
                this.ObstacleCollider = value.GetComponent<Collider>();
            }
        }

        private Collider ObstacleCollider { get; set; }
        public float MaxLookAhead { get; set; }

        public float AvoidMargin { get; set; }

        public float FanAngle { get; set; }

        public DynamicAvoidObstacle(GameObject obstacle)
        {
            
            this.Obstacle = obstacle;
            this.Target = new KinematicData();

            //adjust this values for fine-tuning behavior...
            this.AvoidMargin = 5.0f;
            this.MaxLookAhead = 10.0f;
 
            // For multiple rays. It can also be adjusted...
            this.FanAngle = MathConstants.MATH_PI_4; 
        }

        public override MovementOutput GetMovement()
        {
            if (this.Character.Velocity.sqrMagnitude > 0)  //Raycast does not like zero vector for direction
            {
                RaycastHit hitInfo;
                bool collision = false;

                var color0 = Color.green;

               
                //To create a Ray one uses the Ray Constructor : Ray(Vector3 origin, Vector3 direction);
                //The Raycast method asks for a normalized ray... Note that (Vector3 to normalize).normalized, the current vector is unchanged and a new normalized vector is returned.
                //If you want to normalize the current vector, use Vector3.Normalize(Vector3 to normalize) function.

                Ray r = new Ray(this.Character.Position, Vector3.Normalize(this.Character.Velocity));

                //To get a whisker direction you can rotate the velocity vector, something like 
                // var whisker = MathHelper.Rotate2D(this.Character.velocity, this.FanAngle).normalized;

                //Rays are starting from the center of the car. For more accurate results you can start them in other positions.
                //For instance, the front of the car should be somewhere near var auxPosition = this.Character.Position + normalizedVelocity * 2.0f;


                // Now, how does Unity deals with colisions? Each obstacle has a collider which we put in the property ObstacleCollider...
                // public bool Raycast(Ray ray, out RaycastHit hitInfo, float maxDistance) returns a boolean and the output object hitInfo has information about the collision

                collision = this.ObstacleCollider.Raycast(r, out hitInfo, this.MaxLookAhead);

                if (collision)  //you must add other possibilities to this if when using more than one ray...
                {
                    var hitPoint = hitInfo.point;
                    var hitNormal = hitInfo.normal;
                    color0 = Color.red;
                    //TODO You must give a new target to avoid the obstacle...
                    //this.Target.Position = ...
                    
                    Debug.DrawRay(hitPoint, hitNormal, Color.blue);
                }

                // If you want to draw rays, this is how you do it, make sure you have Gizmo turned on
                Debug.DrawRay(this.Character.Position, Vector3.Normalize(this.Character.Velocity)*this.MaxLookAhead, color0);

                if (collision)
                {
                    return base.GetMovement();
                }

            }


            return new MovementOutput();
        }
    }
}
